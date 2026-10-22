"""
Multivariable Linear Quadratic Integral (LQI) Controller with Anti-Windup.

Designs an optimal state-feedback + integral-action MIMO controller around
the linearized operating point of the Quadruple-Tank Process by solving the
Continuous-Time Algebraic Riccati Equation (CARE) on the augmented state:

    z = [ (x - x_eq)^T ,  xi^T ]^T  in R^6
    d(xi)/dt = r_ref - C_reg * x = [ h_1_ref - h_1 ,  h_2_ref - h_2 ]^T

Control law:
    u = u_eq - K_x * (x_hat - x_eq) - K_i * xi
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import linalg

from quadtank_fdi.plant import QuadrupleTankPlant


@dataclass
class LQIGains:
    """Optimal state-proportional and integral feedback gain matrices."""

    K_x: np.ndarray  # Shape (2, 4)
    K_i: np.ndarray  # Shape (2, 2)
    P_are: np.ndarray  # Shape (6, 6) CARE solution
    closed_loop_poles: np.ndarray  # Eigenvalues of A_aug - B_aug @ K_aug


class MIMO_LQIController:
    """Closed-loop multivariable LQI controller with clamping anti-windup."""

    def __init__(
        self,
        plant: QuadrupleTankPlant,
        u_eq: np.ndarray,
        Q_x: np.ndarray | None = None,
        Q_i: np.ndarray | None = None,
        R_u: np.ndarray | None = None,
        anti_windup_limit: float = 25.0,
    ) -> None:
        self.plant = plant
        self.u_eq = np.asarray(u_eq, dtype=float).copy()
        self.x_eq = self.plant.steady_state_from_input(self.u_eq)
        self.anti_windup_limit = float(anti_windup_limit)

        A_mat, B_mat, _, C_reg = self.plant.linearize(self.x_eq)
        self.C_reg = C_reg

        # Augmented 6-state system: dz/dt = A_aug * z + B_aug * du + [0; I] * dr
        # where d(xi)/dt = r - C_reg * x = -C_reg * dx + dr
        n_x = 4
        n_y = 2
        A_aug = np.zeros((n_x + n_y, n_x + n_y), dtype=float)
        A_aug[:n_x, :n_x] = A_mat
        A_aug[n_x:, :n_x] = -C_reg

        B_aug = np.zeros((n_x + n_y, 2), dtype=float)
        B_aug[:n_x, :] = B_mat

        if Q_x is None:
            Q_x = np.diag([2.5, 2.5, 0.4, 0.4])
        if Q_i is None:
            Q_i = np.diag([0.18, 0.18])
        if R_u is None:
            R_u = np.diag([0.8, 0.8])

        Q_aug = linalg.block_diag(Q_x, Q_i)
        P_are = linalg.solve_continuous_are(A_aug, B_aug, Q_aug, R_u)
        K_aug = linalg.inv(R_u) @ (B_aug.T @ P_are)

        cl_poles = linalg.eigvals(A_aug - B_aug @ K_aug)
        self.gains = LQIGains(
            K_x=K_aug[:, :n_x],
            K_i=K_aug[:, n_x:],
            P_are=P_are,
            closed_loop_poles=cl_poles,
        )
        self.xi = np.zeros(2, dtype=float)

    def reset(self) -> None:
        """Reset integral error accumulator."""
        self.xi = np.zeros(2, dtype=float)

    def compute_control(
        self,
        x_feedback: np.ndarray,
        y_regulated: np.ndarray,
        r_ref: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Compute saturated pump command voltages and update integral state.

        Uses conditional integration (clamping anti-windup) so integral states
        do not wind up when pump actuators hit [u_min, u_max].

        Args:
            x_feedback: Estimated or measured 4-tank state vector [cm].
            y_regulated: Regulated lower-tank levels [y_1, y_2] in [cm].
            r_ref: Desired lower-tank setpoints [r_1, r_2] in [cm].
            dt: Sampling interval [s].

        Returns:
            Commanded pump voltage vector u_cmd in [V], clipped to [u_min, u_max].
        """
        tracking_error = np.asarray(r_ref, dtype=float) - np.asarray(
            y_regulated, dtype=float
        )
        xi_candidate = np.clip(
            self.xi + tracking_error * dt,
            -self.anti_windup_limit,
            self.anti_windup_limit,
        )

        dx = np.asarray(x_feedback, dtype=float) - self.x_eq
        u_unsat = self.u_eq - (self.gains.K_x @ dx) - (self.gains.K_i @ xi_candidate)
        u_cmd = np.clip(u_unsat, self.plant.params.u_min, self.plant.params.u_max)

        # Conditional anti-windup: only integrate channels that are not saturating
        # in the direction that worsens saturation.
        for j in range(2):
            at_upper = (u_unsat[j] > self.plant.params.u_max) and (
                tracking_error[j] > 0.0
            )
            at_lower = (u_unsat[j] < self.plant.params.u_min) and (
                tracking_error[j] < 0.0
            )
            if not (at_upper or at_lower):
                self.xi[j] = xi_candidate[j]

        return u_cmd
