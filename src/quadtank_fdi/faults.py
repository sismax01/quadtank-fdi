"""
First-Principles Fault & Instrument Anomaly Injection Engine.

Supports deterministic and stochastic injection of:
1. Sensor Drift (incipient ramp bias on level transmitters)
2. Sensor Frozen / Stuck Signal (ADC / transmitter lock-up with zero variance)
3. Sensor Step Bias (abrupt calibration offset)
4. Actuator Degradation (multiplicative loss of pump effectiveness / cavitation)
5. Physical Tank Leak (hull or drain-line orifice puncture obeying Torricelli's law)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple
import numpy as np


class FaultCategory(str, Enum):
    """Canonical physical and instrumentation fault categories."""

    NOMINAL = "nominal"
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_FROZEN = "sensor_frozen"
    SENSOR_BIAS = "sensor_bias"
    ACTUATOR_DEGRADATION = "actuator_degradation"
    TANK_LEAK = "tank_leak"


@dataclass(frozen=True)
class FaultProfile:
    """Configuration of a physical or instrumentation fault scenario.

    Attributes:
        category: Type of fault to inject.
        channel: Zero-based index of affected component:
            - 0..3 for Tank/Sensor 1..4
            - 0..1 for Pump Actuator 1..2
        onset_time: Simulation time t_0 [s] when the fault begins.
        magnitude: Physical fault parameter:
            - SENSOR_DRIFT: drift slope beta [cm / s]
            - SENSOR_BIAS: step offset delta_y [cm]
            - SENSOR_FROZEN: unused (holds value at t = onset_time)
            - ACTUATOR_DEGRADATION: fractional loss alpha in (0, 1) [-]
            - TANK_LEAK: effective leak orifice area a_leak [cm^2]
        incipient_time_constant: Optional exponential rise time constant [s]
            for gradual physical fault evolution (0.0 for abrupt onset).
    """

    category: FaultCategory = FaultCategory.NOMINAL
    channel: int = 0
    onset_time: float = 100.0
    magnitude: float = 0.0
    incipient_time_constant: float = 0.0

    @property
    def description(self) -> str:
        """Human-readable engineering description of the fault scenario."""
        if self.category == FaultCategory.NOMINAL:
            return "Nominal Fault-Free Operation"
        if self.category == FaultCategory.SENSOR_DRIFT:
            return (
                f"Sensor {self.channel + 1} Incipient Ramp Drift "
                f"({self.magnitude:+.4f} cm/s @ t={self.onset_time:.0f}s)"
            )
        if self.category == FaultCategory.SENSOR_FROZEN:
            return (
                f"Sensor {self.channel + 1} Frozen Transmitter Signal "
                f"(@ t={self.onset_time:.0f}s)"
            )
        if self.category == FaultCategory.SENSOR_BIAS:
            return (
                f"Sensor {self.channel + 1} Abrupt Calibration Bias "
                f"({self.magnitude:+.2f} cm @ t={self.onset_time:.0f}s)"
            )
        if self.category == FaultCategory.ACTUATOR_DEGRADATION:
            return (
                f"Pump {self.channel + 1} Actuator Loss of Effectiveness "
                f"(-{100.0 * self.magnitude:.1f}% @ t={self.onset_time:.0f}s)"
            )
        if self.category == FaultCategory.TANK_LEAK:
            return (
                f"Tank {self.channel + 1} Physical Orifice Leak "
                f"(a_leak={self.magnitude:.4f} cm^2 @ t={self.onset_time:.0f}s)"
            )
        return f"{self.category.value} (ch={self.channel})"


class FaultInjector:
    """Applies physical plant faults and instrumentation anomalies at runtime."""

    def __init__(self, profile: FaultProfile) -> None:
        self.profile = profile
        self._frozen_value: float | None = None

    def reset(self) -> None:
        """Clear latched state (e.g. frozen sensor reading)."""
        self._frozen_value = None

    def is_active(self, t: float) -> bool:
        """Return True if the fault is active at simulation time t [s]."""
        if self.profile.category == FaultCategory.NOMINAL:
            return False
        return t >= self.profile.onset_time

    def _envelope(self, t: float) -> float:
        """Compute time activation envelope in [0, 1] after onset_time."""
        if not self.is_active(t):
            return 0.0
        dt_active = t - self.profile.onset_time
        tau = self.profile.incipient_time_constant
        if tau <= 1e-6:
            return 1.0
        return float(1.0 - np.exp(-dt_active / tau))

    def apply_process_faults(
        self, t: float, u_cmd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute effective pump input u_eff and tank leak orifice areas a_leak.

        Args:
            t: Current simulation time [s].
            u_cmd: Controller-commanded pump voltage vector [V].

        Returns:
            Tuple of (u_eff [V], leak_areas [cm^2]).
        """
        u_eff = np.asarray(u_cmd, dtype=float).copy()
        leak_areas = np.zeros(4, dtype=float)

        if not self.is_active(t):
            return u_eff, leak_areas

        env = self._envelope(t)
        if self.profile.category == FaultCategory.ACTUATOR_DEGRADATION:
            idx = int(self.profile.channel)
            loss_factor = np.clip(env * self.profile.magnitude, 0.0, 1.0)
            u_eff[idx] = (1.0 - loss_factor) * u_eff[idx]
        elif self.profile.category == FaultCategory.TANK_LEAK:
            idx = int(self.profile.channel)
            leak_areas[idx] = env * max(0.0, self.profile.magnitude)

        return u_eff, leak_areas

    def apply_sensor_faults(
        self, t: float, x_true: np.ndarray, noise_vec: np.ndarray
    ) -> np.ndarray:
        """Compute corrupted sensor measurement vector y_meas from true state x_true.

        Args:
            t: Current simulation time [s].
            x_true: True physical tank levels [h_1..h_4] in [cm].
            noise_vec: Zero-mean Gaussian sensor noise sample in [cm].

        Returns:
            Measured output vector y_meas in [cm].
        """
        y_meas = np.asarray(x_true, dtype=float) + np.asarray(noise_vec, dtype=float)

        if not self.is_active(t):
            return y_meas

        idx = int(self.profile.channel)
        dt_active = t - self.profile.onset_time

        if self.profile.category == FaultCategory.SENSOR_DRIFT:
            y_meas[idx] = y_meas[idx] + self.profile.magnitude * dt_active
        elif self.profile.category == FaultCategory.SENSOR_BIAS:
            env = self._envelope(t)
            y_meas[idx] = y_meas[idx] + env * self.profile.magnitude
        elif self.profile.category == FaultCategory.SENSOR_FROZEN:
            if self._frozen_value is None:
                self._frozen_value = float(x_true[idx] + noise_vec[idx])
            # Frozen sensor outputs a constant locked value with zero thermal noise
            y_meas[idx] = self._frozen_value

        return y_meas


def default_benchmark_scenarios(onset_time: float = 100.0) -> dict[str, FaultProfile]:
    """Return the standard suite of benchmark fault scenarios."""
    return {
        "nominal": FaultProfile(
            category=FaultCategory.NOMINAL,
            channel=0,
            onset_time=onset_time,
            magnitude=0.0,
        ),
        "sensor_drift": FaultProfile(
            category=FaultCategory.SENSOR_DRIFT,
            channel=2,  # Level Sensor 3 drift (+0.025 cm/s -> +5.0 cm over 200s)
            onset_time=onset_time,
            magnitude=0.025,
        ),
        "sensor_frozen": FaultProfile(
            category=FaultCategory.SENSOR_FROZEN,
            channel=1,  # Level Sensor 2 frozen transmitter
            onset_time=onset_time,
            magnitude=0.0,
        ),
        "sensor_bias": FaultProfile(
            category=FaultCategory.SENSOR_BIAS,
            channel=0,  # Level Sensor 1 step calibration bias (+2.0 cm)
            onset_time=onset_time,
            magnitude=2.0,
        ),
        "actuator_degradation": FaultProfile(
            category=FaultCategory.ACTUATOR_DEGRADATION,
            channel=0,  # Pump 1 loses 28% effectiveness (cavitation / impeller wear)
            onset_time=onset_time,
            magnitude=0.28,
            incipient_time_constant=5.0,
        ),
        "tank_leak": FaultProfile(
            category=FaultCategory.TANK_LEAK,
            channel=0,  # Tank 1 orifice leak (a_leak = 0.018 cm^2, ~25% of nominal orifice)
            onset_time=onset_time,
            magnitude=0.018,
            incipient_time_constant=3.0,
        ),
    }
