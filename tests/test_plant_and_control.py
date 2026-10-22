"""
Unit tests for Quadruple-Tank nonlinear physics, analytical linearization,
Relative Gain Array (RGA), Kalman ranks, and closed-loop MIMO LQI control.

Compatible with both `pytest` and standard library `python -m unittest`.
"""

from __future__ import annotations

import unittest
import numpy as np

from quadtank_fdi.controller import MIMO_LQIController
from quadtank_fdi.plant import QuadTankParams, QuadrupleTankPlant


class TestPlantAndControl(unittest.TestCase):
    """Test suite for nonlinear plant dynamics, linearization, and LQI control."""

    def test_steady_state_equilibrium_zero_derivatives(self) -> None:
        """Verify analytical equilibrium h_eq produces identically zero dh/dt."""
        plant = QuadrupleTankPlant()
        u_eq = np.array([3.0, 3.0])
        x_eq = plant.steady_state_from_input(u_eq)

        dh = plant.derivatives(x_eq, u_eq)
        np.testing.assert_allclose(dh, np.zeros(4), atol=1e-12)

        x_next = plant.rk4_step(x_eq, u_eq, dt=0.5)
        np.testing.assert_allclose(x_next, x_eq, atol=1e-12)

    def test_analytical_jacobian_matches_finite_difference(self) -> None:
        """Verify analytical Jacobian A = df/dx matches central finite differences."""
        plant = QuadrupleTankPlant()
        u_eq = np.array([3.2, 2.8])
        x_eq = plant.steady_state_from_input(u_eq)

        A_analytic = plant.jacobian_A(x_eq)
        eps = 1e-6
        A_num = np.zeros((4, 4), dtype=float)
        for j in range(4):
            dx = np.zeros(4, dtype=float)
            dx[j] = eps
            f_plus = plant.derivatives(x_eq + dx, u_eq)
            f_minus = plant.derivatives(x_eq - dx, u_eq)
            A_num[:, j] = (f_plus - f_minus) / (2.0 * eps)

        np.testing.assert_allclose(A_analytic, A_num, rtol=1e-6, atol=1e-8)

    def test_controllability_observability_and_minimum_phase(self) -> None:
        """Verify full rank (rank = 4) for controllability and observability."""
        plant = QuadrupleTankPlant()
        self.assertTrue(plant.params.is_minimum_phase)

        x_eq = plant.steady_state_from_input(np.array([3.0, 3.0]))
        ctrb_rk, obsv_full_rk, obsv_reg_rk = (
            plant.check_controllability_observability(x_eq)
        )
        self.assertEqual(ctrb_rk, 4)
        self.assertEqual(obsv_full_rk, 4)
        self.assertEqual(obsv_reg_rk, 4)

        nmp_params = QuadTankParams(gamma=(0.40, 0.40))
        self.assertFalse(nmp_params.is_minimum_phase)

    def test_rga_row_and_column_sums(self) -> None:
        """Verify Relative Gain Array rows and columns sum to 1.0."""
        plant = QuadrupleTankPlant()
        x_eq = plant.steady_state_from_input(np.array([3.0, 3.0]))
        rga = plant.compute_rga(x_eq)
        np.testing.assert_allclose(np.sum(rga, axis=0), np.ones(2), atol=1e-10)
        np.testing.assert_allclose(np.sum(rga, axis=1), np.ones(2), atol=1e-10)

    def test_lqi_closed_loop_stability_and_step_tracking(self) -> None:
        """Verify CARE closed-loop eigenvalues are strictly stable and track setpoints."""
        plant = QuadrupleTankPlant()
        u_eq = np.array([3.0, 3.0])
        ctrl = MIMO_LQIController(plant, u_eq=u_eq)

        self.assertTrue(np.all(np.real(ctrl.gains.closed_loop_poles) < -1e-3))

        x = ctrl.x_eq.copy()
        r_target = ctrl.x_eq[:2] + np.array([1.5, -1.0])
        dt = 0.5
        for _ in range(400):
            u_cmd = ctrl.compute_control(x, x[:2], r_target, dt)
            x = plant.rk4_step(x, u_cmd, dt)

        np.testing.assert_allclose(x[:2], r_target, atol=0.02)


if __name__ == "__main__":
    unittest.main()
