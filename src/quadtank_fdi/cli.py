"""
Command-Line Interface (CLI) for the Quadruple-Tank Digital Twin & FDI Benchmark.

Usage:
    python -m quadtank_fdi.cli --output-dir artifacts/output
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict

from quadtank_fdi.plotting import plot_benchmark_suite
from quadtank_fdi.simulator import BenchmarkSimulator, SimulationTelemetry


def export_summary_csv(
    results: Dict[str, SimulationTelemetry], csv_path: str | Path
) -> Path:
    """Export quantitative benchmark metrics across all scenarios to CSV."""
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "true_fault_category",
        "true_channel",
        "onset_time_s",
        "detected",
        "detection_time_s",
        "detection_latency_s",
        "isolated_fault_category",
        "isolated_channel",
        "confidence",
        "pre_fault_ekf_rmse_cm",
        "correct_detection",
        "correct_isolation",
        "diagnostic_explanation",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, tel in results.items():
            diag = tel.final_diagnosis
            writer.writerow(
                {
                    "scenario": key,
                    "true_fault_category": tel.fault_profile.category.value,
                    "true_channel": (
                        tel.fault_profile.channel + 1
                        if tel.fault_profile.category.value != "nominal"
                        else "None"
                    ),
                    "onset_time_s": (
                        f"{tel.fault_profile.onset_time:.1f}"
                        if tel.fault_profile.category.value != "nominal"
                        else "N/A"
                    ),
                    "detected": diag.detected,
                    "detection_time_s": (
                        f"{diag.detection_time:.1f}"
                        if diag.detection_time is not None
                        else "N/A"
                    ),
                    "detection_latency_s": (
                        f"{tel.detection_latency:.1f}"
                        if tel.detection_latency is not None
                        else "N/A"
                    ),
                    "isolated_fault_category": diag.isolated_type.value,
                    "isolated_channel": (
                        diag.isolated_channel + 1
                        if diag.isolated_channel is not None
                        else "None"
                    ),
                    "confidence": f"{diag.confidence:.2f}",
                    "pre_fault_ekf_rmse_cm": f"{tel.pre_fault_ekf_rmse:.4f}",
                    "correct_detection": tel.correct_detection,
                    "correct_isolation": tel.correct_isolation,
                    "diagnostic_explanation": diag.explanation,
                }
            )
    return out


def format_console_table(results: Dict[str, SimulationTelemetry]) -> str:
    """Format a crisp engineering summary table for terminal output."""
    lines = []
    header = (
        f"{'Scenario':<22} | {'True Fault (Ch)':<24} | {'Det. Latency':<13} | "
        f"{'Isolated Root Cause (Ch)':<26} | {'EKF RMSE':<10} | {'Verdict':<7}"
    )
    sep = "-" * len(header)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for key, tel in results.items():
        true_ch = (
            f"Ch {tel.fault_profile.channel + 1}"
            if tel.fault_profile.category.value != "nominal"
            else "-"
        )
        true_str = f"{tel.fault_profile.category.value} ({true_ch})"
        lat_str = (
            f"{tel.detection_latency:.1f} s"
            if tel.detection_latency is not None
            else "None (0 FA)"
        )
        iso_ch = (
            f"Ch {tel.final_diagnosis.isolated_channel + 1}"
            if tel.final_diagnosis.isolated_channel is not None
            else "-"
        )
        iso_str = f"{tel.final_diagnosis.isolated_type.value} ({iso_ch})"
        rmse_str = f"{tel.pre_fault_ekf_rmse:.4f} cm"
        verdict = "PASS" if tel.correct_isolation else "FAIL"
        lines.append(
            f"{key:<22} | {true_str:<24} | {lat_str:<13} | "
            f"{iso_str:<26} | {rmse_str:<10} | {verdict:<7}"
        )
    lines.append(sep)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the end-to-end Quadruple-Tank Digital Twin & FDI benchmark."""
    parser = argparse.ArgumentParser(
        description="Run the Quadruple-Tank Process Digital Twin & FDI Benchmark Suite."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save diagnostic plots and CSV telemetry summary.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=250.0,
        help="Simulation duration in seconds (default: 250.0).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.5,
        help="Sampling step size in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--onset",
        type=float,
        default=100.0,
        help="Fault injection onset time in seconds (default: 100.0).",
    )
    args = parser.parse_args(argv)

    sim = BenchmarkSimulator(dt=args.dt, duration=args.duration)
    ctrb_rk, obsv_full_rk, obsv_reg_rk = sim.plant.check_controllability_observability(
        sim.x_eq
    )
    rga = sim.plant.compute_rga(sim.x_eq)

    print("======================================================================")
    print(" QUADRUPLE-TANK PROCESS (JOHANSSON 2000) DIGITAL TWIN & FDI BENCHMARK ")
    print("======================================================================")
    print(f"Equilibrium Pump Input u_eq : {sim.u_eq.round(3)} V")
    print(f"Equilibrium Tank Levels x_eq: {sim.x_eq.round(3)} cm")
    print(
        f"Minimum-Phase Configuration : {sim.plant.params.is_minimum_phase} "
        f"(gamma_1 + gamma_2 = {sum(sim.plant.params.gamma):.2f})"
    )
    print(
        f"Kalman Ranks (n=4)          : Ctrb={ctrb_rk}, Obsv(Full)={obsv_full_rk}, "
        f"Obsv(Lower Only)={obsv_reg_rk}"
    )
    print(f"Steady-State RGA Diagonal   : [{rga[0, 0]:.3f}, {rga[1, 1]:.3f}]")
    print()

    results = sim.run_all_benchmarks(onset_time=args.onset)
    print(format_console_table(results))

    out_dir = Path(args.output_dir)
    fig_path = plot_benchmark_suite(
        results, out_dir / "quad_tank_fdi_diagnostics.png"
    )
    csv_path = export_summary_csv(
        results, out_dir / "quad_tank_fdi_benchmark_metrics.csv"
    )

    print(f"\nSaved diagnostic plot : {fig_path}")
    print(f"Saved metrics summary : {csv_path}")

    all_passed = all(tel.correct_isolation for tel in results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
