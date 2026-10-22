"""
Integration & diagnostic verification tests for the CD-EKF, Structured Observer
Bank, and Fault Detection & Isolation (FDI) pipeline across all fault modes.

Compatible with both `pytest` and standard library `python -m unittest`.
"""

from __future__ import annotations

import unittest

from quadtank_fdi.faults import FaultCategory, FaultProfile
from quadtank_fdi.fdi import IsolatedFaultType
from quadtank_fdi.simulator import BenchmarkSimulator


class TestFDIPipeline(unittest.TestCase):
    """End-to-end verification of the observer bank and FDI isolation engine."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = BenchmarkSimulator(dt=0.5, duration=250.0, seed=42)
        cls.results = cls.sim.run_all_benchmarks(onset_time=100.0)

    def test_nominal_zero_false_alarms_and_ekf_precision(self) -> None:
        """Verify zero false alarms and sub-sensor-noise EKF RMSE in nominal operation."""
        nom = self.results["nominal"]
        self.assertFalse(nom.final_diagnosis.detected)
        self.assertEqual(nom.final_diagnosis.isolated_type, IsolatedFaultType.NONE)
        self.assertTrue(nom.correct_detection)
        self.assertTrue(nom.correct_isolation)
        self.assertLess(nom.pre_fault_ekf_rmse, 0.025)

    def test_canonical_fault_detection_and_isolation(self) -> None:
        """Verify every benchmark fault is accurately detected and isolated to root cause."""
        expectations = [
            ("sensor_drift", IsolatedFaultType.SENSOR_DRIFT, 2, 35.0),
            ("sensor_frozen", IsolatedFaultType.SENSOR_FROZEN, 1, 20.0),
            ("sensor_bias", IsolatedFaultType.SENSOR_BIAS, 0, 5.0),
            ("actuator_degradation", IsolatedFaultType.ACTUATOR_DEGRADATION, 0, 25.0),
            ("tank_leak", IsolatedFaultType.TANK_LEAK, 0, 25.0),
        ]
        for key, exp_type, exp_ch, max_lat in expectations:
            with self.subTest(scenario=key):
                tel = self.results[key]
                self.assertTrue(tel.final_diagnosis.detected)
                self.assertTrue(tel.correct_detection)
                self.assertEqual(tel.final_diagnosis.isolated_type, exp_type)
                self.assertEqual(tel.final_diagnosis.isolated_channel, exp_ch)
                self.assertTrue(tel.correct_isolation)
                self.assertIsNotNone(tel.detection_latency)
                assert tel.detection_latency is not None
                self.assertLessEqual(tel.detection_latency, max_lat)

    def test_pump2_degradation_and_upper_tank3_leak_isolation(self) -> None:
        """Verify isolation generalizes to Pump 2 degradation and Upper Tank 3 cascade leak."""
        sim = BenchmarkSimulator(dt=0.5, duration=250.0, seed=101)

        pump2_profile = FaultProfile(
            category=FaultCategory.ACTUATOR_DEGRADATION,
            channel=1,
            onset_time=100.0,
            magnitude=0.28,
            incipient_time_constant=4.0,
        )
        tel_p2 = sim.run_scenario("pump2_loss", pump2_profile)
        self.assertTrue(tel_p2.correct_isolation)
        self.assertEqual(
            tel_p2.final_diagnosis.isolated_type,
            IsolatedFaultType.ACTUATOR_DEGRADATION,
        )
        self.assertEqual(tel_p2.final_diagnosis.isolated_channel, 1)

        tank3_leak_profile = FaultProfile(
            category=FaultCategory.TANK_LEAK,
            channel=2,
            onset_time=100.0,
            magnitude=0.020,
            incipient_time_constant=2.0,
        )
        tel_t3 = sim.run_scenario("tank3_leak", tank3_leak_profile)
        self.assertTrue(tel_t3.correct_isolation)
        self.assertEqual(
            tel_t3.final_diagnosis.isolated_type, IsolatedFaultType.TANK_LEAK
        )
        self.assertEqual(tel_t3.final_diagnosis.isolated_channel, 2)


if __name__ == "__main__":
    unittest.main()
