"""
First-Principles Nonlinear Continuous-Time Quadruple-Tank Process Model.

Reference:
    K. H. Johansson, "The Quadruple-Tank Process: A Multivariable Laboratory
    Process with an Adjustable Zero," IEEE Transactions on Control Systems
    Technology, vol. 8, no. 3, pp. 456-465, May 2000.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy import linalg


@dataclass(frozen=True)
class QuadTankParams:
    """Physical parameters of the Johansson (2000) Quadruple-Tank Process.

    Attributes:
        A: Cross-sectional areas of Tanks 1..4 [cm^2].
        a: Outlet orifice cross-sectional areas of Tanks 1..4 [cm^2].
        k: Pump flow-per-Volt gains for Pumps 1..2 [cm^3 / (V * s)].
        gamma: Three-way valve split fractions [gamma_1, gamma_2] in (0, 1).
            Pump 1 delivers gamma_1 * k_1 * u_1 to Tank 1 (lower) and
            (1 - gamma_1) * k_1 * u_1 to Tank 4 (upper).
            Pump 2 delivers gamma_2 * k_2 * u_2 to Tank 2 (lower) and
            (1 - gamma_2) * k_2 * u_2 to Tank 3 (upper).
        g: Gravitational acceleration [cm / s^2].
        h_max: Physical overflow height of each tank [cm].
        u_min: Minimum pump command voltage [V].
        u_max: Maximum pump command voltage [V].
    """

    A: Tuple[float, float, float, float] = (28.0, 32.0, 28.0, 32.0)
    a: Tuple[float, float, float, float] = (0.071, 0.057, 0.071, 0.057)
    k: Tuple[float, float] = (3.33, 3.35)
    gamma: Tuple[float, float] = (0.70, 0.60)
    g: float = 981.0
    h_max: float = 35.0
    u_min: float = 0.0
    u_max: float = 10.0

    @property
    def is_minimum_phase(self) -> bool:
        """System is minimum-phase iff 1 < gamma_1 + gamma_2 < 2."""
        return 1.0 < (self.gamma[0] + self.gamma[1]) < 2.0


class QuadrupleTankPlant:
    """Continuous-time nonlinear dynamical model of the Quadruple-Tank Process.

    State vector:
        x = [h_1, h_2, h_3, h_4]^T  (liquid levels in [cm])
    Control input vector:
        u = [u_1, u_2]^T            (pump voltages in [V])
    Measured output vector:
        y = [h_1, h_2, h_3, h_4]^T  (full level transmitter suite in [cm])
    Regulated output vector:
        y_reg = [h_1, h_2]^T        (lower tank levels tracked by LQI controller)
    """

    def __init__(self, params: QuadTankParams | None = None) -> None:
        self.params = params if params is not None else QuadTankParams()
        self.A = np.asarray(self.params.A, dtype=float)
        self.a = np.asarray(self.params.a, dtype=float)
        self.k = np.asarray(self.params.k, dtype=float)
        self.gamma = np.asarray(self.params.gamma, dtype=float)
        self.g = float(self.params.g)

    def steady_state_from_input(self, u_eq: np.ndarray) -> np.ndarray:
        """Compute exact analytical equilibrium levels h_eq for constant input u_eq.

        By setting dh_i/dt = 0 in Bernoulli's orifice outflow equations:
            a_3 * sqrt(2 g h_3) = (1 - gamma_2) * k_2 * u_2
            a_4 * sqrt(2 g h_4) = (1 - gamma_1) * k_1 * u_1
            a_1 * sqrt(2 g h_1) = a_3 * sqrt(2 g h_3) + gamma_1 * k_1 * u_1
            a_2 * sqrt(2 g h_2) = a_4 * sqrt(2 g h_4) + gamma_2 * k_2 * u_2
        """
        u1, u2 = float(u_eq[0]), float(u_eq[1])
        A1, A2, A3, A4 = self.A
        a1, a2, a3, a4 = self.a
        k1, k2 = self.k
        gam1, gam2 = self.gamma
        g = self.g

        q_in_3 = (1.0 - gam2) * k2 * u2
        q_in_4 = (1.0 - gam1) * k1 * u1
        q_in_1 = q_in_3 + gam1 * k1 * u1
        q_in_2 = q_in_4 + gam2 * k2 * u2

        h1_eq = (q_in_1 / a1) ** 2 / (2.0 * g)
        h2_eq = (q_in_2 / a2) ** 2 / (2.0 * g)
        h3_eq = (q_in_3 / a3) ** 2 / (2.0 * g)
        h4_eq = (q_in_4 / a4) ** 2 / (2.0 * g)

        return np.array([h1_eq, h2_eq, h3_eq, h4_eq], dtype=float)

    def derivatives(
        self,
        x: np.ndarray,
        u: np.ndarray,
        leak_areas: np.ndarray | None = None,
        process_disturbance: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate continuous-time state derivatives dx/dt = f(x, u, d).

        Args:
            x: State vector [h_1, h_2, h_3, h_4] in [cm].
            u: Effective pump voltages [u_1, u_2] in [V].
            leak_areas: Optional orifice leak area per tank [a_leak_1..4] in [cm^2].
                Each leak drains fluid at rate a_leak_i * sqrt(2 * g * h_i).
            process_disturbance: Optional additive volumetric flow disturbance
                per tank [cm^3 / s] or level rate disturbance [cm / s].

        Returns:
            Time derivative vector dh/dt in [cm / s].
        """
        h = np.maximum(np.asarray(x, dtype=float), 1e-6)
        u_eff = np.maximum(np.asarray(u, dtype=float), 0.0)
        a_leak = (
            np.zeros(4, dtype=float)
            if leak_areas is None
            else np.maximum(np.asarray(leak_areas, dtype=float), 0.0)
        )

        sqrt_2gh = np.sqrt(2.0 * self.g * h)
        outflow = (self.a + a_leak) * sqrt_2gh  # Total gravity outflow [cm^3/s]
        upper_outflow_3 = self.a[2] * sqrt_2gh[2]
        upper_outflow_4 = self.a[3] * sqrt_2gh[3]

        gam1, gam2 = self.gamma
        k1, k2 = self.k
        u1, u2 = u_eff[0], u_eff[1]

        dh = np.zeros(4, dtype=float)
        dh[0] = (-outflow[0] + upper_outflow_3 + gam1 * k1 * u1) / self.A[0]
        dh[1] = (-outflow[1] + upper_outflow_4 + gam2 * k2 * u2) / self.A[1]
        dh[2] = (-outflow[2] + (1.0 - gam2) * k2 * u2) / self.A[2]
        dh[3] = (-outflow[3] + (1.0 - gam1) * k1 * u1) / self.A[3]

        if process_disturbance is not None:
            dh = dh + np.asarray(process_disturbance, dtype=float)

        return dh

    def rk4_step(
        self,
        x: np.ndarray,
        u: np.ndarray,
        dt: float,
        leak_areas: np.ndarray | None = None,
        process_disturbance: np.ndarray | None = None,
    ) -> np.ndarray:
        """Integrate continuous-time nonlinear dynamics over [t, t + dt] via RK4."""
        k1 = self.derivatives(x, u, leak_areas, process_disturbance)
        k2 = self.derivatives(x + 0.5 * dt * k1, u, leak_areas, process_disturbance)
        k3 = self.derivatives(x + 0.5 * dt * k2, u, leak_areas, process_disturbance)
        k4 = self.derivatives(x + dt * k3, u, leak_areas, process_disturbance)
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return np.clip(x_next, 1e-4, self.params.h_max)

    def jacobian_A(self, x: np.ndarray, leak_areas: np.ndarray | None = None) -> np.ndarray:
        """Compute exact analytical state Jacobian F = df/dx at state x."""
        h = np.maximum(np.asarray(x, dtype=float), 1e-4)
        a_leak = (
            np.zeros(4, dtype=float)
            if leak_areas is None
            else np.maximum(np.asarray(leak_areas, dtype=float), 0.0)
        )
        # Time constants T_i = (A_i / a_i) * sqrt(2 * h_i / g)
        # Derivative of (a_i * sqrt(2 * g * h_i)) w.r.t. h_i is a_i * g / sqrt(2 * g * h_i)
        d_sqrt = self.g / np.sqrt(2.0 * self.g * h)

        A_mat = np.zeros((4, 4), dtype=float)
        A_mat[0, 0] = -((self.a[0] + a_leak[0]) / self.A[0]) * d_sqrt[0]
        A_mat[0, 2] = (self.a[2] / self.A[0]) * d_sqrt[2]

        A_mat[1, 1] = -((self.a[1] + a_leak[1]) / self.A[1]) * d_sqrt[1]
        A_mat[1, 3] = (self.a[3] / self.A[1]) * d_sqrt[3]

        A_mat[2, 2] = -((self.a[2] + a_leak[2]) / self.A[2]) * d_sqrt[2]
        A_mat[3, 3] = -((self.a[3] + a_leak[3]) / self.A[3]) * d_sqrt[3]
        return A_mat

    def jacobian_B(self) -> np.ndarray:
        """Compute exact analytical input matrix B = df/du."""
        gam1, gam2 = self.gamma
        k1, k2 = self.k
        A1, A2, A3, A4 = self.A
        return np.array(
            [
                [gam1 * k1 / A1, 0.0],
                [0.0, gam2 * k2 / A2],
                [0.0, (1.0 - gam2) * k2 / A3],
                [(1.0 - gam1) * k1 / A4, 0.0],
            ],
            dtype=float,
        )

    def linearize(
        self, x_eq: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return continuous-time linearized matrices (A, B, C_full, C_reg) at x_eq."""
        A_mat = self.jacobian_A(x_eq)
        B_mat = self.jacobian_B()
        C_full = np.eye(4, dtype=float)
        C_reg = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        return A_mat, B_mat, C_full, C_reg

    def compute_rga(self, x_eq: np.ndarray) -> np.ndarray:
        """Compute the steady-state Relative Gain Array (RGA) Lambda = G(0) .* G(0)^-T."""
        A_mat, B_mat, _, C_reg = self.linearize(x_eq)
        G0 = -C_reg @ linalg.inv(A_mat) @ B_mat
        return G0 * linalg.inv(G0).T

    def check_controllability_observability(
        self, x_eq: np.ndarray
    ) -> Tuple[int, int, int]:
        """Compute Kalman controllability and observability ranks at x_eq.

        Returns:
            Tuple of (controllability_rank, full_observability_rank,
                      lower_tanks_only_observability_rank).
        """
        A_mat, B_mat, C_full, C_reg = self.linearize(x_eq)
        n = A_mat.shape[0]
        ctrb_blocks = [B_mat]
        obsv_full_blocks = [C_full]
        obsv_reg_blocks = [C_reg]
        A_power = np.eye(n)
        for _ in range(1, n):
            A_power = A_power @ A_mat
            ctrb_blocks.append(A_power @ B_mat)
            obsv_full_blocks.append(C_full @ A_power)
            obsv_reg_blocks.append(C_reg @ A_power)

        ctrb_mat = np.hstack(ctrb_blocks)
        obsv_full_mat = np.vstack(obsv_full_blocks)
        obsv_reg_mat = np.vstack(obsv_reg_blocks)

        return (
            int(np.linalg.matrix_rank(ctrb_mat)),
            int(np.linalg.matrix_rank(obsv_full_mat)),
            int(np.linalg.matrix_rank(obsv_reg_mat)),
        )
