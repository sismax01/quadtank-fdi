"""
First-Principles State Estimation & Structured Observer Bank for FDI.

Implements:
1. ContinuousDiscreteEKF:
   Full-state Continuous-Discrete Extended Kalman Filter propagating nonlinear
   Bernoulli tank dynamics via RK4, analytical state Jacobians via exact matrix
   exponential discretization, Joseph-form covariance updates, and Normalized
   Innovation Squared (NIS) statistics.

2. StructuredObserverBank:
   Combines:
   - The full-order CD-EKF for minimum-variance state estimation and innovation
     monitoring.
   - An input-driven Parallel Reference Observer (open-loop nonlinear digital
     twin synchronized to the plant operating point) whose parity residual
     e(t) = y(t) - x_par(t) is strictly invariant to closed-loop controller
     compensation during sensor faults.
   - A Cascade Consistency Residual generator comparing upper-to-lower tank
     Bernoulli gravity inflow predicted from measured upper levels vs. parallel
     modelled upper levels.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import linalg

from quadtank_fdi.plant import QuadrupleTankPlant


@dataclass
class ObserverStepOutput:
    """Instantaneous state estimates and structured residuals at step k."""

    x_ekf_post: np.ndarray        # Posterior EKF state estimate x_hat(k|k) [cm]
    x_ekf_prior: np.ndarray       # Prior EKF state estimate x_hat(k|k-1) [cm]
    x_parallel: np.ndarray        # Input-driven parallel model state x_par(k) [cm]
    innovation: np.ndarray        # EKF innovation r(k) = y(k) - x_hat(k|k-1) [cm]
    whitened_innovation: np.ndarray  # S(k)^(-1/2) * r(k) [-]
    nis: float                    # Normalized Innovation Squared r^T S^-1 r [-]
    parity_residual: np.ndarray   # Parallel parity residual e(k) = y(k) - x_par(k) [cm]
    parity_rate: np.ndarray       # Filtered derivative de/dt(k) [cm/s]
    orifice_balance_residual: np.ndarray  # Dynamic ODE residual de/dt - A_ii * e [cm/s]
    cascade_residual: np.ndarray  # Lower-tank cascade consistency residual [cm]


class ContinuousDiscreteEKF:
    """Continuous-Discrete Extended Kalman Filter (CD-EKF) for the 4-Tank Plant."""

    def __init__(
        self,
        plant: QuadrupleTankPlant,
        x0: np.ndarray,
        P0: np.ndarray | None = None,
        Q_cov: np.ndarray | None = None,
        R_cov: np.ndarray | None = None,
    ) -> None:
        self.plant = plant
        self.x_hat = np.asarray(x0, dtype=float).copy()
        self.P = (
            np.eye(4, dtype=float) * 0.05
            if P0 is None
            else np.asarray(P0, dtype=float).copy()
        )
        # Continuous-time process noise spectral density Q_c and measurement noise R
        self.Q = (
            np.eye(4, dtype=float) * 1e-4
            if Q_cov is None
            else np.asarray(Q_cov, dtype=float).copy()
        )
        self.R = (
            np.eye(4, dtype=float) * (0.03**2)
            if R_cov is None
            else np.asarray(R_cov, dtype=float).copy()
        )
        self.H = np.eye(4, dtype=float)

    def reset(self, x0: np.ndarray) -> None:
        """Reset state estimate and covariance to initial conditions."""
        self.x_hat = np.asarray(x0, dtype=float).copy()
        self.P = np.eye(4, dtype=float) * 0.05

    def step(
        self, u_cmd: np.ndarray, y_meas: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Execute one continuous-time prediction and discrete measurement update.

        Args:
            u_cmd: Known commanded pump voltage vector [V].
            y_meas: Measured 4-tank level vector [cm].
            dt: Sampling step [s].

        Returns:
            Tuple of (x_post, x_prior, innovation, whitened_innovation, nis).
        """
        # 1. Nonlinear state propagation via 4th-order Runge-Kutta
        x_prior = self.plant.rk4_step(self.x_hat, u_cmd, dt)

        # 2. Linearized state transition matrix Phi_k = exp(F(x_hat) * dt)
        F_jac = self.plant.jacobian_A(self.x_hat)
        Phi = linalg.expm(F_jac * dt)
        P_prior = Phi @ self.P @ Phi.T + self.Q * dt

        # 3. Innovation & Innovation Covariance S_k
        innovation = np.asarray(y_meas, dtype=float) - (self.H @ x_prior)
        S = self.H @ P_prior @ self.H.T + self.R
        S_inv = linalg.inv(S)
        S_sqrt_inv = linalg. fractional_matrix_power(S, -0.5).real

        # 4. Optimal Kalman Gain & Posterior Update (Joseph stabilized form)
        K = P_prior @ self.H.T @ S_inv
        x_post = np.clip(x_prior + K @ innovation, 1e-4, self.plant.params.h_max)
        I_KH = np.eye(4) - K @ self.H
        P_post = I_KH @ P_prior @ I_KH.T + K @ self.R @ K.T

        self.x_hat = x_post
        self.P = 0.5 * (P_post + P_post.T)

        whitened = S_sqrt_inv @ innovation
        nis = float(innovation.T @ S_inv @ innovation)
        return x_post, x_prior, innovation, whitened, nis


class StructuredObserverBank:
    """Bank of observers generating directional & dynamic Analytical Redundancy Relations.

    Combines:
    1. Full-state CD-EKF (`ekf`) for state estimation and NIS calculation.
    2. Input-driven Parallel Reference Model (`x_parallel`) for closed-loop-invariant
       parity residuals e(k) = y(k) - x_parallel(k).
    3. Cascade Lower-Tank Observer (`x_cascade_lower`) driven by measured upper-tank
       levels (y_3, y_4) instead of modelled upper-tank levels (x_par_3, x_par_4).
       When Sensor 3 or 4 drifts, x_cascade_lower diverges while e_1 or e_2 stays zero!
    4. First-order low-pass differentiator on e(k) to evaluate the dynamic orifice
       mass-balance relation: mu_i(t) = de_i/dt - A_ii * e_i(t).
    """

    def __init__(
        self,
        plant: QuadrupleTankPlant,
        x0: np.ndarray,
        sigma_meas: float = 0.03,
        diff_filter_tau: float = 8.0,
    ) -> None:
        self.plant = plant
        self.sigma_meas = float(sigma_meas)
        self.diff_filter_tau = float(diff_filter_tau)

        R_cov = np.eye(4) * (sigma_meas**2)
        Q_cov = np.eye(4) * 1e-4
        self.ekf = ContinuousDiscreteEKF(plant, x0=x0, Q_cov=Q_cov, R_cov=R_cov)

        self.x_parallel = np.asarray(x0, dtype=float).copy()
        self.x_cascade = np.asarray(x0, dtype=float).copy()
        self._prev_parity = np.zeros(4, dtype=float)
        self._parity_rate = np.zeros(4, dtype=float)

    def reset(self, x0: np.ndarray) -> None:
        """Reset all observers in the bank to initial equilibrium x0."""
        self.ekf.reset(x0)
        self.x_parallel = np.asarray(x0, dtype=float).copy()
        self.x_cascade = np.asarray(x0, dtype=float).copy()
        self._prev_parity = np.zeros(4, dtype=float)
        self._parity_rate = np.zeros(4, dtype=float)

    def step(
        self, u_cmd: np.ndarray, y_meas: np.ndarray, dt: float
    ) -> ObserverStepOutput:
        """Advance all observers by one sample interval dt [s]."""
        y = np.asarray(y_meas, dtype=float)
        u = np.asarray(u_cmd, dtype=float)

        # 1. Full CD-EKF step
        x_post, x_prior, innov, whitened, nis = self.ekf.step(u, y, dt)

        # 2. Input-driven Parallel Reference Model step
        self.x_parallel = self.plant.rk4_step(self.x_parallel, u, dt)
        parity_res = y - self.x_parallel

        # 3. Cascade Lower-Tank Observer step:
        # Uses measured upper-tank levels y[2], y[3] to drive lower tanks 1 & 2
        x_cas_in = self.x_cascade.copy()
        x_cas_in[2] = max(y[2], 1e-4)
        x_cas_in[3] = max(y[3], 1e-4)
        x_cas_next = self.plant.rk4_step(x_cas_in, u, dt)
        self.x_cascade[0] = x_cas_next[0]
        self.x_cascade[1] = x_cas_next[1]
        self.x_cascade[2] = self.x_parallel[2]
        self.x_cascade[3] = self.x_parallel[3]

        # Cascade discrepancy between cascade-predicted lower levels and parallel model
        cascade_res = np.zeros(4, dtype=float)
        cascade_res[0] = y[0] - self.x_cascade[0]
        cascade_res[1] = y[1] - self.x_cascade[1]
        cascade_res[2] = self.x_cascade[0] - self.x_parallel[0]
        cascade_res[3] = self.x_cascade[1] - self.x_parallel[1]

        # 4. Filtered time derivative of parity residual de/dt
        raw_rate = (parity_res - self._prev_parity) / max(dt, 1e-6)
        alpha = dt / (self.diff_filter_tau + dt)
        self._parity_rate = (1.0 - alpha) * self._parity_rate + alpha * raw_rate
        self._prev_parity = parity_res.copy()

        # 5. Dynamic orifice mass-balance residual: mu_i = de_i/dt - A_ii * e_i
        F_diag = np.diag(self.plant.jacobian_A(self.x_parallel))
        orifice_balance = self._parity_rate - F_diag * parity_res

        return ObserverStepOutput(
            x_ekf_post=x_post,
            x_ekf_prior=x_prior,
            x_parallel=self.x_parallel.copy(),
            innovation=innov,
            whitened_innovation=whitened,
            nis=nis,
            parity_residual=parity_res,
            parity_rate=self._parity_rate.copy(),
            orifice_balance_residual=orifice_balance,
            cascade_residual=cascade_res,
        )
