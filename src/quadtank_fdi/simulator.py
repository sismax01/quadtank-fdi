"""
Closed-Loop Digital Twin & FDI Benchmark Simulation Engine.

Executes end-to-end continuous-time RK4 plant integration, MIMO LQI feedback
control, stochastic sensor/process noise generation, physical/instrument fault
injection, CD-EKF + Structured Observer Bank estimation, and real-time CUSUM/GLRT
fault detection and isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

from quadtank_fdi.controller import MIMO_LQIController
from quadtank_fdi.faults import (
    FaultCategory,
    FaultInjector,
    FaultProfile,
    default_benchmark_scenarios,
)
from quadtank_fdi.fdi import DiagnosisResult, FDIDiagnosticEngine, IsolatedFaultType
from quadtank_fdi.observers import StructuredObserverBank
from quadtank_fdi.plant import QuadTankParams, QuadrupleTankPlant


@dataclass
class SimulationTelemetry:
    """Complete time-series telemetry and diagnostic performance metrics."""

    scenario_name: str
    fault_profile: FaultProfile
    t: np.ndarray                 # (N,) time vector [s]
    x_true: np.ndarray            # (N, 4) true physical tank levels [cm]
    y_meas: np.ndarray            # (N, 4) measured sensor levels [cm]
    x_ekf: np.ndarray             # (N, 4) CD-EKF posterior state estimates [cm]
    x_parallel: np.ndarray        # (N, 4) parallel reference model states [cm]
    r_ref: np.ndarray             # (N, 2) regulated lower-tank setpoints [cm]
    u_cmd: np.ndarray             # (N, 2) controller-commanded pump voltages [V]
    u_eff: np.ndarray             # (N, 2) effective physical pump voltages [V]
    innovation: np.ndarray        # (N, 4) CD-EKF innovations [cm]
    nis: np.ndarray               # (N,) Normalized Innovation Squared [-]
    parity_residual: np.ndarray   # (N, 4) parallel model parity residuals [cm]
    cusum_stat: np.ndarray        # (N, 4) per-channel CUSUM statistics [-]
    glrt_stat: np.ndarray         # (N,) global GLRT statistics [-]
    variance_ratio: np.ndarray    # (N, 4) sliding-window sensor variance ratios [-]
    detected_trace: np.ndarray    # (N,) boolean alarm trace
    final_diagnosis: DiagnosisResult

    @property
    def detection_latency(self) -> float | None:
        """Detection delay t_detect - t_onset [s], or None if not detected / nominal."""
        if (
            self.fault_profile.category == FaultCategory.NOMINAL
            or not self.final_diagnosis.detected
            or self.final_diagnosis.detection_time is None
        ):
            return None
        return float(
            max(0.0, self.final_diagnosis.detection_time - self.fault_profile.onset_time)
        )

    @property
    def pre_fault_ekf_rmse(self) -> float:
        """Root-mean-square state estimation error [cm] prior to fault onset."""
        mask = self.t < self.fault_profile.onset_time
        if not np.any(mask):
            mask = np.ones_like(self.t, dtype=bool)
        err = self.x_ekf[mask] - self.x_true[mask]
        return float(np.sqrt(np.mean(err**2)))

    @property
    def correct_detection(self) -> bool:
        """True if detection matches ground-truth fault presence without false alarm."""
        if self.fault_profile.category == FaultCategory.NOMINAL:
            return not self.final_diagnosis.detected
        return (
            self.final_diagnosis.detected
            and self.final_diagnosis.detection_time is not None
            and self.final_diagnosis.detection_time >= self.fault_profile.onset_time
        )

    @property
    def correct_isolation(self) -> bool:
        """True if both fault category and component channel match ground truth."""
        if self.fault_profile.category == FaultCategory.NOMINAL:
            return (
                not self.final_diagnosis.detected
                and self.final_diagnosis.isolated_type == IsolatedFaultType.NONE
            )
        return (
            self.correct_detection
            and self.final_diagnosis.isolated_type.value
            == self.fault_profile.category.value
            and self.final_diagnosis.isolated_channel == self.fault_profile.channel
        )


class BenchmarkSimulator:
    """Closed-loop simulation runner for the Quadruple-Tank FDI benchmark."""

    def __init__(
        self,
        params: QuadTankParams | None = None,
        u_eq: np.ndarray | None = None,
        dt: float = 0.5,
        duration: float = 250.0,
        sigma_meas: float = 0.03,
        sigma_proc: float = 0.002,
        seed: int = 42,
    ) -> None:
        self.plant = QuadrupleTankPlant(params)
        self.u_eq = (
            np.array([3.0, 3.0], dtype=float)
            if u_eq is None
            else np.asarray(u_eq, dtype=float)
        )
        self.x_eq = self.plant.steady_state_from_input(self.u_eq)
        self.dt = float(dt)
        self.duration = float(duration)
        self.sigma_meas = float(sigma_meas)
        self.sigma_proc = float(sigma_proc)
        self.seed = int(seed)

    def setpoint_schedule(self, t: float) -> np.ndarray:
        """Return regulated lower-tank level setpoints [r_1(t), r_2(t)] in [cm].

        Includes a +1.0 cm / -0.8 cm dynamic setpoint transition at t = 25 s to
        verify that closed-loop reference tracking does not trigger false FDI alarms.
        """
        r = self.x_eq[:2].copy()
        if t >= 25.0:
            r[0] += 1.0
            r[1] -= 0.8
        return r

    def run_scenario(
        self,
        scenario_name: str,
        fault_profile: FaultProfile,
        seed: int | None = None,
    ) -> SimulationTelemetry:
        """Execute a complete closed-loop simulation for a given FaultProfile."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        n_steps = int(round(self.duration / self.dt)) + 1
        t_vec = np.linspace(0.0, self.duration, n_steps)

        controller = MIMO_LQIController(self.plant, u_eq=self.u_eq)
        observer_bank = StructuredObserverBank(
            self.plant, x0=self.x_eq, sigma_meas=self.sigma_meas
        )
        fdi_engine = FDIDiagnosticEngine(self.plant, sigma_meas=self.sigma_meas)
        fault_injector = FaultInjector(fault_profile)

        # Pre-allocate telemetry arrays
        x_true_arr = np.zeros((n_steps, 4), dtype=float)
        y_meas_arr = np.zeros((n_steps, 4), dtype=float)
        x_ekf_arr = np.zeros((n_steps, 4), dtype=float)
        x_par_arr = np.zeros((n_steps, 4), dtype=float)
        r_ref_arr = np.zeros((n_steps, 2), dtype=float)
        u_cmd_arr = np.zeros((n_steps, 2), dtype=float)
        u_eff_arr = np.zeros((n_steps, 2), dtype=float)
        innov_arr = np.zeros((n_steps, 4), dtype=float)
        nis_arr = np.zeros(n_steps, dtype=float)
        parity_arr = np.zeros((n_steps, 4), dtype=float)
        cusum_arr = np.zeros((n_steps, 4), dtype=float)
        glrt_arr = np.zeros(n_steps, dtype=float)
        var_ratio_arr = np.ones((n_steps, 4), dtype=float)
        detected_arr = np.zeros(n_steps, dtype=bool)

        x_curr = self.x_eq.copy()
        u_cmd = self.u_eq.copy()
        last_diag: DiagnosisResult | None = None

        for k, t in enumerate(t_vec):
            r_ref = self.setpoint_schedule(t)
            meas_noise = rng.normal(0.0, self.sigma_meas, size=4)
            y_meas = fault_injector.apply_sensor_faults(t, x_curr, meas_noise)

            # Observer bank update & residual generation
            obs_out = observer_bank.step(u_cmd, y_meas, self.dt)

            # FDI evaluation
            diag = fdi_engine.evaluate(t, y_meas, obs_out, self.dt)
            last_diag = diag

            # Closed-loop LQI control law using measured regulated levels
            u_cmd = controller.compute_control(
                x_feedback=obs_out.x_ekf_post,
                y_regulated=y_meas[:2],
                r_ref=r_ref,
                dt=self.dt,
            )

            # Apply physical actuator / tank leak faults
            u_eff, leak_areas = fault_injector.apply_process_faults(t, u_cmd)

            # Record step telemetry
            x_true_arr[k] = x_curr
            y_meas_arr[k] = y_meas
            x_ekf_arr[k] = obs_out.x_ekf_post
            x_par_arr[k] = obs_out.x_parallel
            r_ref_arr[k] = r_ref
            u_cmd_arr[k] = u_cmd
            u_eff_arr[k] = u_eff
            innov_arr[k] = obs_out.innovation
            nis_arr[k] = obs_out.nis
            parity_arr[k] = obs_out.parity_residual
            cusum_arr[k] = diag.cusum_stat
            glrt_arr[k] = diag.glrt_stat
            var_ratio_arr[k] = diag.variance_ratio
            detected_arr[k] = diag.detected

            # Integrate true physical plant to k+1 with slight process noise
            proc_dist = rng.normal(0.0, self.sigma_proc, size=4)
            x_curr = self.plant.rk4_step(
                x_curr,
                u_eff,
                self.dt,
                leak_areas=leak_areas,
                process_disturbance=proc_dist,
            )

        assert last_diag is not None
        return SimulationTelemetry(
            scenario_name=scenario_name,
            fault_profile=fault_profile,
            t=t_vec,
            x_true=x_true_arr,
            y_meas=y_meas_arr,
            x_ekf=x_ekf_arr,
            x_parallel=x_par_arr,
            r_ref=r_ref_arr,
            u_cmd=u_cmd_arr,
            u_eff=u_eff_arr,
            innovation=innov_arr,
            nis=nis_arr,
            parity_residual=parity_arr,
            cusum_stat=cusum_arr,
            glrt_stat=glrt_arr,
            variance_ratio=var_ratio_arr,
            detected_trace=detected_arr,
            final_diagnosis=last_diag,
        )

    def run_all_benchmarks(
        self, onset_time: float = 100.0
    ) -> Dict[str, SimulationTelemetry]:
        """Run all 6 canonical benchmark scenarios and return their telemetry dict."""
        scenarios = default_benchmark_scenarios(onset_time=onset_time)
        results: Dict[str, SimulationTelemetry] = {}
        for idx, (name, profile) in enumerate(scenarios.items()):
            results[name] = self.run_scenario(
                scenario_name=name, fault_profile=profile, seed=self.seed + idx
            )
        return results
