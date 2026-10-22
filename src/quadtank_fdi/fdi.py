"""
Model-Based Fault Detection and Isolation (FDI) Diagnostic Engine.

Combines:
1. Two-Sided Page-Hinkley / Cumulative Sum (CUSUM) sequential change detection
   and Generalized Likelihood Ratio Test (GLRT) on normalized parity & innovation
   residuals.
2. Sliding-Window Sample Variance Ratio Detector for frozen/stuck instrumentation
   (`SENSOR_FROZEN`).
3. Directed-Graph Structural & Dynamical Fault Signature Isolation:
   - Pump 1 Degradation: simultaneously drives Tank 1 (e_1 < 0) and Upper Tank 4
     (e_4 < 0, cascading to Tank 2), while Upper Tank 3 remains unaffected (e_3 ~ 0).
   - Pump 2 Degradation: simultaneously drives Tank 2 (e_2 < 0) and Upper Tank 3
     (e_3 < 0, cascading to Tank 1), while Upper Tank 4 remains unaffected (e_4 ~ 0).
   - Upper Tank 3 Leak: drives Tank 3 (e_3 < 0) and Lower Tank 1 (e_1 < 0) with
     e_2 ~ 0 and e_4 ~ 0.
   - Upper Tank 4 Leak: drives Tank 4 (e_4 < 0) and Lower Tank 2 (e_2 < 0) with
     e_1 ~ 0 and e_3 ~ 0.
   - Lower Tank 1 or 2 Leak vs. Sensor Drift vs. Sensor Step Bias: isolated via
     single-channel confinement, hydraulic step-rate bounds, and Bernoulli orifice
     exponential damping.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List
import numpy as np

from quadtank_fdi.observers import ObserverStepOutput
from quadtank_fdi.plant import QuadrupleTankPlant


class IsolatedFaultType(str, Enum):
    """Root-cause classification emitted by the FDI isolation logic."""

    NONE = "nominal"
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_FROZEN = "sensor_frozen"
    SENSOR_BIAS = "sensor_bias"
    ACTUATOR_DEGRADATION = "actuator_degradation"
    TANK_LEAK = "tank_leak"


@dataclass
class DiagnosisResult:
    """Instantaneous and latched diagnostic output at time step k."""

    detected: bool
    detection_time: float | None
    isolated_type: IsolatedFaultType
    isolated_channel: int | None
    cusum_stat: np.ndarray          # Per-channel two-sided CUSUM statistic [-]
    glrt_stat: float                # Global GLRT chi-square statistic [-]
    variance_ratio: np.ndarray      # Sliding-window dy variance ratio per sensor [-]
    channel_flags: np.ndarray       # Boolean array of active residual channels (4,)
    confidence: float               # Diagnostic isolation confidence in [0, 1]
    explanation: str = ""


@dataclass
class FDIDiagnosticEngine:
    """First-principles residual evaluation, sequential detection, and fault isolation."""

    plant: QuadrupleTankPlant
    sigma_meas: float = 0.03
    cusum_drift_allowance: float = 1.8     # Slack parameter k_cusum in sigma units
    cusum_threshold: float = 22.0          # Decision threshold h_cusum
    parity_z_threshold: float = 4.5        # Normalized residual magnitude threshold
    freeze_window: int = 25                # Sliding window length [samples]
    freeze_var_ratio_threshold: float = 0.04  # Variance collapse threshold

    # Internal state
    _s_pos: np.ndarray = field(init=False, repr=False)
    _s_neg: np.ndarray = field(init=False, repr=False)
    _dy_buffers: List[Deque[float]] = field(init=False, repr=False)
    _post_detect_parity: List[List[float]] = field(init=False, repr=False)
    _prev_y: np.ndarray | None = field(default=None, init=False, repr=False)
    _max_innov_jump: np.ndarray = field(init=False, repr=False)
    _detected: bool = field(default=False, init=False, repr=False)
    _detection_time: float | None = field(default=None, init=False, repr=False)
    _steps_since_detection: int = field(default=0, init=False, repr=False)
    _latched_type: IsolatedFaultType = field(
        default=IsolatedFaultType.NONE, init=False, repr=False
    )
    _latched_channel: int | None = field(default=None, init=False, repr=False)
    _latched_explanation: str = field(default="Nominal", init=False, repr=False)
    _latched_confidence: float = field(default=1.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all sequential detectors, sliding windows, and latched diagnoses."""
        self._s_pos = np.zeros(4, dtype=float)
        self._s_neg = np.zeros(4, dtype=float)
        self._dy_buffers = [deque(maxlen=self.freeze_window) for _ in range(4)]
        self._post_detect_parity = [[] for _ in range(4)]
        self._prev_y = None
        self._max_innov_jump = np.zeros(4, dtype=float)
        self._detected = False
        self._detection_time = None
        self._steps_since_detection = 0
        self._latched_type = IsolatedFaultType.NONE
        self._latched_channel = None
        self._latched_explanation = (
            "Nominal operation (all residuals within 3-sigma bounds)."
        )
        self._latched_confidence = 1.0

    def evaluate(
        self,
        t: float,
        y_meas: np.ndarray,
        obs_out: ObserverStepOutput,
        dt: float,
    ) -> DiagnosisResult:
        """Process one step of observer telemetry and update detection/isolation state."""
        y = np.asarray(y_meas, dtype=float)
        parity = obs_out.parity_residual

        # 1. Update first-difference buffers for frozen-sensor variance monitoring
        if self._prev_y is not None:
            dy = y - self._prev_y
            for i in range(4):
                self._dy_buffers[i].append(float(dy[i]))
                step_jump = abs(float(dy[i]))
                if step_jump > self._max_innov_jump[i]:
                    self._max_innov_jump[i] = step_jump
        self._prev_y = y.copy()

        # Compute variance ratio Var(dy_i) / (2 * sigma_meas^2)
        expected_dy_var = max(2.0 * (self.sigma_meas**2), 1e-10)
        var_ratio = np.ones(4, dtype=float)
        for i in range(4):
            if len(self._dy_buffers[i]) >= self.freeze_window:
                arr = np.asarray(self._dy_buffers[i], dtype=float)
                var_ratio[i] = float(np.var(arr) / expected_dy_var)

        # 2. Two-sided CUSUM on normalized parity residuals z_i = e_i / sigma_meas
        z_parity = parity / max(self.sigma_meas, 1e-6)
        self._s_pos = np.maximum(
            0.0, self._s_pos + z_parity - self.cusum_drift_allowance
        )
        self._s_neg = np.maximum(
            0.0, self._s_neg - z_parity - self.cusum_drift_allowance
        )
        cusum_stat = np.maximum(self._s_pos, self._s_neg)

        # Global GLRT statistic (sum of squared normalized parity residuals)
        glrt_stat = float(np.sum(z_parity**2))

        # Active channel flags (CUSUM exceeded or frozen sensor variance collapsed)
        freeze_flags = var_ratio < self.freeze_var_ratio_threshold
        cusum_flags = (cusum_stat > self.cusum_threshold) & (
            np.abs(z_parity) > self.parity_z_threshold
        )
        channel_flags = cusum_flags | freeze_flags

        # 3. Fault Detection Trigger
        if not self._detected and np.any(channel_flags):
            self._detected = True
            self._detection_time = float(t)
            self._steps_since_detection = 0

        # 4. Fault Isolation Logic
        if self._detected:
            self._steps_since_detection += 1
            for i in range(4):
                self._post_detect_parity[i].append(float(parity[i]))
            self._isolate_fault(
                parity, z_parity, cusum_stat, var_ratio, freeze_flags, dt
            )

        return DiagnosisResult(
            detected=self._detected,
            detection_time=self._detection_time,
            isolated_type=self._latched_type,
            isolated_channel=self._latched_channel,
            cusum_stat=cusum_stat.copy(),
            glrt_stat=glrt_stat,
            variance_ratio=var_ratio,
            channel_flags=channel_flags,
            confidence=self._latched_confidence,
            explanation=self._latched_explanation,
        )

    def _isolate_fault(
        self,
        parity: np.ndarray,
        z_parity: np.ndarray,
        cusum_stat: np.ndarray,
        var_ratio: np.ndarray,
        freeze_flags: np.ndarray,
        dt: float,
    ) -> None:
        """Apply directed-graph Fault Signature Matrix (FSM) & dynamic ODE rules."""
        # Rule 1: Frozen / Stuck Sensor Isolation (Variance Ratio Test)
        if np.any(freeze_flags):
            ch = int(np.argmin(var_ratio))
            self._latched_type = IsolatedFaultType.SENSOR_FROZEN
            self._latched_channel = ch
            self._latched_confidence = 0.99
            self._latched_explanation = (
                f"Isolated SENSOR_FROZEN on Level Transmitter {ch + 1}: "
                f"high-frequency innovation variance ratio collapsed to "
                f"{var_ratio[ch]:.4f} (< {self.freeze_var_ratio_threshold})."
            )
            return

        # Rule 2: Multi-Channel Input-Coupled Actuator Degradation
        # Pump 1 feeds Lower Tank 1 (idx 0) and Upper Tank 4 (idx 3, which cascades to Tank 2).
        # Upper Tank 3 (idx 2) is completely decoupled from Pump 1 (e_3 ~ 0).
        pump1_coupled = (
            (z_parity[0] < -4.0)
            and (z_parity[3] < -2.5 or self._s_neg[3] > 6.0)
            and (abs(z_parity[2]) < 3.5)
        )
        if pump1_coupled:
            self._latched_type = IsolatedFaultType.ACTUATOR_DEGRADATION
            self._latched_channel = 0
            self._latched_confidence = 0.97
            self._latched_explanation = (
                "Isolated ACTUATOR_DEGRADATION on Pump 1: simultaneous negative "
                "parity residuals on coupled Tanks 1 & 4 matching B[:,0] input "
                "split while decoupled Upper Tank 3 remains nominal."
            )
            return

        # Pump 2 feeds Lower Tank 2 (idx 1) and Upper Tank 3 (idx 2, which cascades to Tank 1).
        # Upper Tank 4 (idx 3) is completely decoupled from Pump 2 (e_4 ~ 0).
        pump2_coupled = (
            (z_parity[1] < -4.0 or z_parity[2] < -4.0)
            and (z_parity[1] < -2.5 or self._s_neg[1] > 6.0)
            and (z_parity[2] < -2.5 or self._s_neg[2] > 6.0)
            and (abs(z_parity[3]) < 3.5)
        )
        if pump2_coupled:
            self._latched_type = IsolatedFaultType.ACTUATOR_DEGRADATION
            self._latched_channel = 1
            self._latched_confidence = 0.97
            self._latched_explanation = (
                "Isolated ACTUATOR_DEGRADATION on Pump 2: simultaneous negative "
                "parity residuals on coupled Tanks 2 & 3 matching B[:,1] input "
                "split while decoupled Upper Tank 4 remains nominal."
            )
            return

        # Rule 3: Upper-to-Lower Gravity Cascade Leak Isolation
        # A physical leak in Upper Tank 3 causes negative residuals in Tank 3 AND Tank 1,
        # while both Tank 2 and Tank 4 remain at zero residual.
        tank3_cascade_leak = (
            (z_parity[2] < -4.0)
            and (z_parity[0] < -2.5 or self._s_neg[0] > 5.0)
            and (abs(z_parity[1]) < 3.5)
            and (abs(z_parity[3]) < 3.5)
        )
        if tank3_cascade_leak:
            self._latched_type = IsolatedFaultType.TANK_LEAK
            self._latched_channel = 2
            self._latched_confidence = 0.95
            self._latched_explanation = (
                "Isolated TANK_LEAK in Upper Tank 3: negative mass-balance "
                "residual in Tank 3 propagating via gravity cascade into Lower Tank 1."
            )
            return

        # A physical leak in Upper Tank 4 causes negative residuals in Tank 4 AND Tank 2,
        # while both Tank 1 and Tank 3 remain at zero residual.
        tank4_cascade_leak = (
            (z_parity[3] < -4.0)
            and (z_parity[1] < -2.5 or self._s_neg[1] > 5.0)
            and (abs(z_parity[0]) < 3.5)
            and (abs(z_parity[2]) < 3.5)
        )
        if tank4_cascade_leak:
            self._latched_type = IsolatedFaultType.TANK_LEAK
            self._latched_channel = 3
            self._latched_confidence = 0.95
            self._latched_explanation = (
                "Isolated TANK_LEAK in Upper Tank 4: negative mass-balance "
                "residual in Tank 4 propagating via gravity cascade into Lower Tank 2."
            )
            return

        # Rule 4: Single-Channel Anomaly Isolation (Sensor Bias vs Sensor Drift vs Lower Tank Leak)
        ch = int(np.argmax(cusum_stat))

        # 4a: Abrupt Step Bias detection: single-sample jump far exceeds max physical flow rate
        physical_step_limit = max(0.65, 0.35 * dt + 8.0 * self.sigma_meas)
        if self._max_innov_jump[ch] > physical_step_limit:
            self._latched_type = IsolatedFaultType.SENSOR_BIAS
            self._latched_channel = ch
            self._latched_confidence = 0.98
            self._latched_explanation = (
                f"Isolated SENSOR_BIAS on Level Transmitter {ch + 1}: "
                f"discontinuous single-sample step jump ({self._max_innov_jump[ch]:.2f} cm) "
                f"exceeds physical hydraulic rate limit ({physical_step_limit:.2f} cm)."
            )
            return

        # 4b: Upper tank single-channel anomaly (Tank 3 or Tank 4 alone without lower tank drop)
        if ch in (2, 3):
            downstream = 0 if ch == 2 else 1
            if abs(z_parity[downstream]) < 3.5 or parity[ch] > 0.0:
                self._latched_type = IsolatedFaultType.SENSOR_DRIFT
                self._latched_channel = ch
                self._latched_confidence = 0.96
                self._latched_explanation = (
                    f"Isolated SENSOR_DRIFT on Upper Level Transmitter {ch + 1}: "
                    f"ramp residual confined to Sensor {ch + 1} with zero downstream "
                    f"gravity-cascade perturbation in Lower Tank {downstream + 1}."
                )
                return

        # 4c: Positive residual on any single tank (orifice leaks can only drain fluid, e_i < 0)
        if parity[ch] > 0.0:
            self._latched_type = IsolatedFaultType.SENSOR_DRIFT
            self._latched_channel = ch
            self._latched_confidence = 0.95
            self._latched_explanation = (
                f"Isolated SENSOR_DRIFT on Level Transmitter {ch + 1}: "
                f"positive ramp residual (+{parity[ch]:.2f} cm) is physically "
                f"inconsistent with orifice leakage."
            )
            return

        # 4d: Negative single-channel residual on Lower Tank (ch in {0, 1}):
        # Distinguish Physical Tank Leak (first-order exponential deceleration:
        # de/dt = A_ii * e - q_leak/A -> |de/dt| decays as |e| grows) from
        # Negative Sensor Ramp Drift (constant slope de/dt = beta).
        post_e = np.asarray(self._post_detect_parity[ch], dtype=float)
        if len(post_e) >= 24:
            half = len(post_e) // 2
            slope_early = (post_e[half] - post_e[0]) / max(half * dt, 1e-6)
            slope_late = (post_e[-1] - post_e[half]) / max(
                (len(post_e) - 1 - half) * dt, 1e-6
            )
            if abs(slope_late) < 0.90 * abs(slope_early) or abs(slope_early) > 0.045:
                self._latched_type = IsolatedFaultType.TANK_LEAK
                self._latched_channel = ch
                self._latched_confidence = 0.94
                self._latched_explanation = (
                    f"Isolated TANK_LEAK in Lower Tank {ch + 1}: negative single-tank "
                    f"mass-balance residual exhibiting first-order Bernoulli orifice "
                    f"damping (early rate {slope_early:+.4f} cm/s -> late rate {slope_late:+.4f} cm/s)."
                )
                return
            else:
                self._latched_type = IsolatedFaultType.SENSOR_DRIFT
                self._latched_channel = ch
                self._latched_confidence = 0.92
                self._latched_explanation = (
                    f"Isolated SENSOR_DRIFT on Level Transmitter {ch + 1}: "
                    f"constant-slope linear residual drift ({slope_late:+.4f} cm/s) "
                    f"without orifice damping."
                )
                return

        if ch in (0, 1) and parity[ch] < -4.0 * self.sigma_meas:
            self._latched_type = IsolatedFaultType.TANK_LEAK
            self._latched_channel = ch
            self._latched_confidence = 0.90
            self._latched_explanation = (
                f"Isolated TANK_LEAK in Lower Tank {ch + 1}: negative mass-balance "
                f"residual ({parity[ch]:+.2f} cm) consistent with unmodelled orifice outflow."
            )
        else:
            self._latched_type = IsolatedFaultType.SENSOR_DRIFT
            self._latched_channel = ch
            self._latched_confidence = 0.90
            self._latched_explanation = (
                f"Isolated SENSOR_DRIFT on Level Transmitter {ch + 1}: "
                f"persistent single-channel ramp drift ({parity[ch]:+.2f} cm)."
            )
