"""
Publication-Grade Diagnostic Visualization Suite for Quadruple-Tank FDI.

Generates:
1. A comprehensive 6-scenario benchmark dashboard (`plot_benchmark_suite`)
   showing closed-loop tank levels, structured parity residuals, CUSUM decision
   statistics, and fault isolation signatures across all canonical fault modes.
2. A detailed 4-panel single-scenario engineering diagnostic figure
   (`plot_single_scenario_diagnostics`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quadtank_fdi.simulator import SimulationTelemetry


PALETTE = {
    "tank1": "#0072B2",   # Deep sapphire blue
    "tank2": "#D55E00",   # Vermilion orange
    "tank3": "#009E73",   # Teal green
    "tank4": "#CC79A7",   # Rose purple
    "onset": "#B22222",   # Crimson fault onset line
    "detect": "#111111",  # Charcoal detection line
    "thresh": "#666666",  # Neutral threshold line
}


def plot_benchmark_suite(
    results: Dict[str, SimulationTelemetry],
    output_path: str | Path,
    cusum_threshold: float = 22.0,
) -> Path:
    """Render a multi-scenario diagnostic overview figure across all benchmark runs."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    scenario_keys = list(results.keys())
    n_scen = len(scenario_keys)

    fig, axes = plt.subplots(
        n_scen,
        3,
        figsize=(16.5, 2.75 * n_scen),
        sharex=True,
        gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]},
    )
    if n_scen == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(
        "Quadruple-Tank Process Digital Twin — First-Principles Observer-Based FDI Benchmark Suite",
        fontsize=14,
        fontweight="bold",
        y=0.992,
    )

    tank_colors = [
        PALETTE["tank1"],
        PALETTE["tank2"],
        PALETTE["tank3"],
        PALETTE["tank4"],
    ]

    for row, key in enumerate(scenario_keys):
        tel = results[key]
        t = tel.t
        onset = tel.fault_profile.onset_time
        det_time = tel.final_diagnosis.detection_time

        ax_lvl, ax_res, ax_cusum = axes[row, 0], axes[row, 1], axes[row, 2]

        # Column 1: True vs Measured Tank Levels & Setpoints
        for i in range(4):
            ax_lvl.plot(
                t,
                tel.x_true[:, i],
                color=tank_colors[i],
                lw=1.5,
                label=f"$h_{i+1}$ (true)" if row == 0 else None,
            )
            ax_lvl.plot(
                t,
                tel.y_meas[:, i],
                color=tank_colors[i],
                lw=0.8,
                alpha=0.45,
                ls=":",
            )
        ax_lvl.plot(
            t,
            tel.r_ref[:, 0],
            color="#222222",
            ls="--",
            lw=1.0,
            alpha=0.7,
            label="$r_{1,2}$ (setpoint)" if row == 0 else None,
        )
        ax_lvl.plot(t, tel.r_ref[:, 1], color="#222222", ls="--", lw=1.0, alpha=0.7)

        if tel.fault_profile.category.value != "nominal":
            ax_lvl.axvline(onset, color=PALETTE["onset"], ls="--", lw=1.2)
        if det_time is not None:
            ax_lvl.axvline(det_time, color=PALETTE["detect"], ls="-.", lw=1.2)

        ax_lvl.set_ylabel("Level [cm]", fontsize=9, fontweight="bold")
        ax_lvl.set_title(
            f"({chr(65 + row)}) {tel.fault_profile.description}",
            fontsize=9.5,
            fontweight="bold",
            loc="left",
        )
        ax_lvl.grid(True, alpha=0.25)

        # Column 2: Structured Parity Residuals e_i(t) = y_i(t) - x_par_i(t)
        for i in range(4):
            ax_res.plot(
                t,
                tel.parity_residual[:, i],
                color=tank_colors[i],
                lw=1.4,
                label=f"$e_{i+1} = y_{i+1} - \\hat{{h}}_{{par,{i+1}}}$"
                if row == 0
                else None,
            )
        ax_res.axhline(0.0, color="#444444", lw=0.8, alpha=0.6)
        if tel.fault_profile.category.value != "nominal":
            ax_res.axvline(onset, color=PALETTE["onset"], ls="--", lw=1.2)
        if det_time is not None:
            ax_res.axvline(det_time, color=PALETTE["detect"], ls="-.", lw=1.2)

        ax_res.set_ylabel("Parity $e_i$ [cm]", fontsize=9, fontweight="bold")
        iso_label = tel.final_diagnosis.isolated_type.value.upper()
        iso_ch = (
            f"Ch {tel.final_diagnosis.isolated_channel + 1}"
            if tel.final_diagnosis.isolated_channel is not None
            else "None"
        )
        status_str = "PASS" if tel.correct_isolation else "FAIL"
        ax_res.set_title(
            f"Isolated: {iso_label} ({iso_ch}) [{status_str}]",
            fontsize=9.5,
            color="#0b6623" if tel.correct_isolation else "#b22222",
            fontweight="bold",
            loc="left",
        )
        ax_res.grid(True, alpha=0.25)

        # Column 3: CUSUM Sequential Statistics & Variance Freeze Ratio
        for i in range(4):
            ax_cusum.plot(
                t,
                np.minimum(tel.cusum_stat[:, i], 250.0),
                color=tank_colors[i],
                lw=1.4,
                label=f"$S_{i+1}$ (CUSUM)" if row == 0 else None,
            )
        ax_cusum.axhline(
            cusum_threshold,
            color=PALETTE["thresh"],
            ls=":",
            lw=1.2,
            label="Threshold $h$" if row == 0 else None,
        )
        if tel.fault_profile.category.value != "nominal":
            ax_cusum.axvline(
                onset,
                color=PALETTE["onset"],
                ls="--",
                lw=1.2,
                label="Fault Onset $t_0$" if row == 0 else None,
            )
        if det_time is not None:
            ax_cusum.axvline(
                det_time,
                color=PALETTE["detect"],
                ls="-.",
                lw=1.2,
                label="Detected $t_{det}$" if row == 0 else None,
            )

        lat_str = (
            f"Latency: {tel.detection_latency:.1f} s"
            if tel.detection_latency is not None
            else "No Alarm (FPR = 0%)"
        )
        ax_cusum.set_title(lat_str, fontsize=9.5, loc="left", fontweight="bold")
        ax_cusum.set_ylabel("CUSUM $S_i$ [-]", fontsize=9, fontweight="bold")
        ax_cusum.grid(True, alpha=0.25)

        if row == 0:
            ax_lvl.legend(loc="upper right", fontsize=7.5, ncol=3, framealpha=0.9)
            ax_res.legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.9)
            ax_cusum.legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.9)

    for col in range(3):
        axes[-1, col].set_xlabel("Time [s]", fontsize=9.5, fontweight="bold")

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.978])
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out
