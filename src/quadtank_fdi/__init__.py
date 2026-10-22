"""
quadtank_fdi: First-Principles Digital Twin & Observer-Based FDI Framework.

Implements the nonlinear Johansson (2000) Quadruple-Tank Process, MIMO LQI
state-feedback control with anti-windup, realistic physical & instrumentation
fault injection, a Continuous-Discrete Extended Kalman Filter (CD-EKF) &
structured observer bank, and GLRT/CUSUM fault detection and isolation.
"""

from quadtank_fdi.plant import QuadTankParams, QuadrupleTankPlant
from quadtank_fdi.controller import MIMO_LQIController
from quadtank_fdi.faults import FaultCategory, FaultProfile, FaultInjector
from quadtank_fdi.observers import ContinuousDiscreteEKF, StructuredObserverBank
from quadtank_fdi.fdi import FDIDiagnosticEngine, DiagnosisResult, IsolatedFaultType
from quadtank_fdi.simulator import BenchmarkSimulator, SimulationTelemetry

__all__ = [
    "QuadTankParams",
    "QuadrupleTankPlant",
    "MIMO_LQIController",
    "FaultCategory",
    "FaultProfile",
    "FaultInjector",
    "ContinuousDiscreteEKF",
    "StructuredObserverBank",
    "FDIDiagnosticEngine",
    "DiagnosisResult",
    "IsolatedFaultType",
    "BenchmarkSimulator",
    "SimulationTelemetry",
]

__version__ = "1.0.0"
