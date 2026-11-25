"""
plots.py

All visualizations for the V2G simulation report.
Figures produced:
- ev_load_curves_2500kW_min40soc.png
- voltage_profiles_2500kW_min40soc.png
- voltage_heatmap_2500kW_min40soc.png
- metrics_barchart_2500kW_min40soc.png
- pv_and_load_2500kW_min40soc.png
- total_load_curves_2500kW_min40soc.png
- ev_delta_2500kW_min40soc.png
- soc_profile_2500kW_min40soc.png
- net_grid_power_2500kW_min40soc.png

All figures saved to results/.
"""

from __future__ import annotations

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

STRATS = ["G2V", "RB-V2G", "MPC-V2G"]
COLORS_STRAT = {"G2V": "#E53935", "RB-V2G": "#FB8C00", "MPC-V2G": "#1E88E5"}
LINESTYLES = {"G2V": "--", "RB-V2G": "-.", "MPC-V2G": "-"}
LABELS = {"G2V": "G2V (uncontrolled)", "RB-V2G": "V2G rule-based", "MPC-V2G": "V2G QP-MPC"}
VIOL_LOW, VIOL_HIGH = 0.95, 1.05

# Base load profile (MW total network), index 0 = 00:00. Duplicated here so
# plots.py is self-contained; main.py passes the same array via plot_all().
_BASE_MW_DEFAULT = np.array(
    [
        0.8,
        0.7,
        0.65,
        0.6,
        0.6,
        0.65,
        0.8,
        1.0,
        1.3,
        1.5,
        1.6,
        1.65,
        1.7,
        1.65,
        1.6,
        1.5,
        1.6,
        1.8,
        2.0,
        2.1,
        2.0,
        1.8,
        1.5,
        1.1,
    ]
)


def _time_labels(sim_start_h: int = 6) -> list[str]:
    return [f"{(sim_start_h + s) % 24:02d}:00" for s in range(24)]


def _find_result(results: list, penetration: float, strategy: str):
    for r in results:
        if r["penetration"] == penetration and r["strategy"] == strategy:
            return r
    return None


def _base_load_sim(base_load_total_mw: np.ndarray, sim_start_h: int) -> np.ndarray:
    """Return base load [kW] aligned to simulation time axis."""
    return np.roll(base_load_total_mw, -sim_start_h) * 1000  # MW -> kW


# Figure 1: EV-only load curves

def plot_ev_load_curves(results: list, sim_start_h: int = 6):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    xlabels = _time_labels(sim_start_h)
    slots = np.arange(24)

    for ax, pen in zip(axes, [0.2, 0.5, 0.8]):
        for strat in STRATS:
            r = _find_result(results, pen, strat)
            if r is None:
                continue
            ax.plot(
                slots,
                r["ev_kw"],
                color=COLORS_STRAT[strat],
                linestyle=LINESTYLES[strat],
                linewidth=2,
                label=strat,
            )
            ax.fill_between(slots, r["ev_kw"], alpha=0.08, color=COLORS_STRAT[strat])

        ax.set_title(f"EV penetration {int(pen * 100)}%")
        ax.set_xticks(slots[::3])
        ax.set_xticklabels(xlabels[::3], rotation=45, ha="right")
        ax.set_xlabel("Hour")
        ax.set_ylabel("EV fleet power [kW]")
        ax.legend()

    fig.suptitle("EV Fleet Load Profiles - G2V vs RB-V2G vs MPC-V2G (PV 2500 kW, SOC min 40%)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("results/ev_load_curves_2500kW_min40soc.png", bbox_inches="tight")
    plt.close()
    print("  Saved: results/ev_load_curves_2500kW_min40soc.png")


# Figure 2: Voltage profiles on worst bus

def plot_voltage_profiles(results: list, sim_start_h: int = 6):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    xlabels = _time_labels(sim_start_h)
    slots = np.arange(24)

    for ax, pen in zip(axes, [0.2, 0.5, 0.8]):
        for strat in STRATS:
            r = _find_result(results, pen, strat)
            if r is None:
                continue
            v_matrix = r["voltages"]
            worst_bus = int(np.nanargmin(v_matrix.mean(axis=0)))
            v_worst = v_matrix[:, worst_bus]
            ax.plot(
                slots,
                v_worst,
                color=COLORS_STRAT[strat],
                linestyle=LINESTYLES[strat],
                linewidth=2,
                label=f"{strat} (bus {worst_bus})",
            )

        ax.axhspan(VIOL_LOW, VIOL_HIGH, alpha=0.06, color="green")
        ax.axhline(VIOL_LOW, color="green", linewidth=1, linestyle=":")
        ax.axhline(VIOL_HIGH, color="green", linewidth=1, linestyle=":")
        ax.set_title(f"EV penetration {int(pen * 100)}%")
        ax.set_xticks(slots[::3])
        ax.set_xticklabels(xlabels[::3], rotation=45, ha="right")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Voltage [pu]")
        ax.set_ylim(0.88, 1.06)
        ax.legend(fontsize=8)

    fig.suptitle("Voltage Profile on Worst Bus - G2V vs RB-V2G vs MPC-V2G (PV 2500 kW, SOC min 40%)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("results/voltage_profiles_2500kW_min40soc.png", bbox_inches="tight")
    plt.close()
    print("  Saved: results/voltage_profiles_2500kW_min40soc.png")


# Figure 3: Voltage heatmap

def plot_voltage_heatmaps(results: list, sim_start_h: int = 6):
    pens = [0.2, 0.5, 0.8]
    strats = STRATS
    fig, axes = plt.subplots(len(strats), len(pens), figsize=(16, 11), sharex=True, sharey=True)
    xlabels = _time_labels(sim_start_h)
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0.88, vmax=1.05)

    for row, strat in enumerate(strats):
        for col, pen in enumerate(pens):
            ax = axes[row][col]
            r = _find_result(results, pen, strat)
            if r is None:
                ax.set_visible(False)
                continue
            v_matrix = r["voltages"].T
            ax.imshow(v_matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_title(f"{strat} | EV {int(pen * 100)}%", fontsize=9)
            ax.set_xticks(range(0, 24, 3))
            ax.set_xticklabels(xlabels[::3], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(0, 33, 4))
            ax.set_yticklabels([f"Bus {i}" for i in range(0, 33, 4)], fontsize=7)
            ax.contour(v_matrix, levels=[VIOL_LOW], colors="red", linewidths=0.8, linestyles="--")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, label="Voltage [pu]")
    fig.suptitle("Voltage Heatmap - Bus x Hour (PV 2500 kW, SOC min 40%)", fontsize=12)
    plt.savefig("results/voltage_heatmap_2500kW_min40soc.png", bbox_inches="tight")
    plt.close()
    print("  Saved: results/voltage_heatmap_2500kW_min40soc.png")


# Figure 4: Metrics bar chart

def plot_metrics_barchart(results: list):
    labels, volt_devs, violations, ev_peaks = [], [], [], []

    for pen in [0.2, 0.5, 0.8]:
        for strat in STRATS:
            r = _find_result(results, pen, strat)
            if r is None:
                continue
            v_matrix = r["voltages"]
            labels.append(f"EV{int(pen * 100)}%\n{strat}")
            volt_devs.append(float(np.nanmean(np.abs(v_matrix - 1.0))))
            violations.append(int(np.sum((v_matrix < VIOL_LOW) | (v_matrix > VIOL_HIGH))))
            ev_peaks.append(float(r["ev_kw"].max()))

    x = np.arange(len(labels))
    cols = [COLORS_STRAT[l.split("\n")[1]] for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(x, volt_devs, color=cols, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Mean Voltage Deviation [pu]")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("|V - 1.0| mean [pu]")

    axes[1].bar(x, violations, color=cols, edgecolor="white", linewidth=0.5)
    axes[1].set_title("Voltage Violations [bus-hour count]")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("# (slot, bus) outside [0.95-1.05]")

    axes[2].bar(x, ev_peaks, color=cols, edgecolor="white", linewidth=0.5)
    axes[2].set_title("EV Fleet Peak Load [kW]")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel("Peak power [kW]")

    legend_els = [mpatches.Patch(facecolor=COLORS_STRAT[k], label=k) for k in STRATS]
    fig.legend(handles=legend_els, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Summary Metrics - All Scenarios (PV 2500 kW, SOC min 40%)", fontsize=13, y=1.06)
    plt.tight_layout()
    plt.savefig("results/metrics_barchart_2500kW_min40soc.png", bbox_inches="tight")
    plt.close()
    print("  Saved: results/metrics_barchart_2500kW_min40soc.png")


# Figure 5: PV output vs base load

def plot_pv_and_load(pv_sim: np.ndarray, base_load_total_mw: np.ndarray, sim_start_h: int = 6):
    slots = np.arange(24)
    xlabels = _time_labels(sim_start_h)
    base_sim_kw = _base_load_sim(base_load_total_mw, sim_start_h)
    surplus = np.maximum(pv_sim - base_sim_kw, 0)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(slots, pv_sim, alpha=0.35, color="gold", label="PV output [kW]")
    ax.fill_between(slots, base_sim_kw, alpha=0.25, color="steelblue", label="Base load [kW]")
    ax.plot(slots, pv_sim, color="orange", linewidth=2)
    ax.plot(slots, base_sim_kw, color="steelblue", linewidth=2)
    ax.fill_between(
        slots,
        base_sim_kw,
        pv_sim,
        where=(surplus > 0),
        alpha=0.5,
        color="limegreen",
        label="PV surplus (V2G opportunity)",
    )
    ax.set_xticks(slots[::2])
    ax.set_xticklabels(xlabels[::2], rotation=45, ha="right")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power [kW]")
    ax.set_title("PV Generation vs Base Load - Xian, 2024-06-21 (PV 2500 kW, SOC min 40%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/pv_and_load_2500kW_min40soc.png", bbox_inches="tight")
    plt.close()
    print("  Saved: results/pv_and_load_2500kW_min40soc.png")


# Figure 6: Total load curves (base + EV)

def plot_total_load_curves(results: list, base_load_total_mw: np.ndarray, sim_start_h: int = 6,
                           pv_sim: np.ndarray | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    xlabels = _time_labels(sim_start_h)
    slots = np.arange(24)
    base_sim_kw = _base_load_sim(base_load_total_mw, sim_start_h)

    for ax, pen in zip(axes, [0.2, 0.5, 0.8]):
        if pv_sim is not None:
            ax.fill_between(slots, pv_sim, alpha=0.30, color="gold", zorder=0, label="PV generation")
            ax.plot(slots, pv_sim, color="orange", linewidth=1.2, zorder=0)
        ax.fill_between(slots, base_sim_kw, alpha=0.2, color="grey")
        ax.plot(slots, base_sim_kw, color="grey", linewidth=1.5, linestyle=":", label="Base load only")

        for strat in STRATS:
            r = _find_result(results, pen, strat)
            if r is None:
                continue
            total_kw = base_sim_kw + r["ev_kw"]
            ax.plot(
                slots,
                total_kw,
                color=COLORS_STRAT[strat],
                linestyle=LINESTYLES[strat],
                linewidth=2.5,
                label=f"Base + {strat}",
            )
            ax.fill_between(slots, base_sim_kw, total_kw, alpha=0.12, color=COLORS_STRAT[strat])

        r_g2v = _find_result(results, pen, "G2V")
        if r_g2v:
            peak_g2v = float((base_sim_kw + r_g2v["ev_kw"]).max())
            lines = []
            for strat in ("RB-V2G", "MPC-V2G"):
                r = _find_result(results, pen, strat)
                if r:
                    lines.append(f"{strat}: peak down {peak_g2v - float((base_sim_kw + r['ev_kw']).max()):.0f} kW")
            ax.annotate(
                "\n".join(lines),
                xy=(0.03, 0.97),
                xycoords="axes fraction",
                ha="left",
                va="top",
                fontsize=8,
                color="darkgreen",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )

        ax.set_title(f"EV penetration {int(pen * 100)}%")
        ax.set_xticks(slots[::3])
        ax.set_xticklabels(xlabels[::3], rotation=45, ha="right")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Total network load [kW]")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)

    fig.suptitle("Total Load Curve - Base Load + EV (G2V vs RB-V2G vs MPC-V2G, PV 2500 kW, SOC min 40%)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("results/total_load_curves_2500kW_min40soc.png", bbox_inches="tight")
    plt.close()
    print("  Saved: results/total_load_curves_2500kW_min40soc.png")


# Figure 7: EV delta (charge/discharge contribution)

def plot_ev_delta(results: list, sim_start_h: int = 6):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    xlabels = _time_labels(sim_start_h)
    slots = np.arange(24)
    bar_w = 0.27

    for ax, pen in zip(axes, [0.2, 0.5, 0.8]):
        for off, strat in zip((-bar_w, 0.0, bar_w), STRATS):
            r = _find_result(results, pen, strat)
            if r is None:
                continue
            ev = r["ev_kw"]
            ax.bar(slots + off, np.maximum(ev, 0), width=bar_w, color=COLORS_STRAT[strat],
                   alpha=0.85, label=f"{strat} (charge)")
            ax.bar(slots + off, np.minimum(ev, 0), width=bar_w, color=COLORS_STRAT[strat],
                   alpha=0.85, hatch="//", edgecolor="white", label=f"{strat} (inject)")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"EV penetration {int(pen * 100)}%")
        ax.set_xticks(slots[::3])
        ax.set_xticklabels(xlabels[::3], rotation=45, ha="right")
        ax.set_xlabel("Hour")
        ax.set_ylabel("EV power [kW] (negative = injection)")
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("EV Contribution per Slot - G2V vs RB-V2G vs MPC-V2G (PV 2500 kW, SOC min 40%)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig("results/ev_delta_2500kW_min40soc.png", bbox_inches="tight")
    plt.close()
    print("  Saved: results/ev_delta_2500kW_min40soc.png")


# Figure 8: SOC profile

def plot_soc_profile(results: list, sim_start_h: int = 6):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    xlabels = _time_labels(sim_start_h)
    slots = np.arange(24)

    for ax, pen in zip(axes, [0.2, 0.5, 0.8]):
        for strat in STRATS:
            r = _find_result(results, pen, strat)
            if r is None:
                continue
            soc = r.get("soc_profile")
            if soc is None:
                continue
            ax.plot(
                slots,
                soc * 100.0,
                color=COLORS_STRAT[strat],
                linestyle=LINESTYLES[strat],
                linewidth=2,
                label=strat,
            )

        ax.set_title(f"EV penetration {int(pen * 100)}%")
        ax.set_xticks(slots[::3])
        ax.set_xticklabels(xlabels[::3], rotation=45, ha="right")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Average SOC [%]")
        ax.axhline(40, color="orange", linestyle="--", linewidth=1.5, label="SOC min 40%")
        ax.set_ylim(30, 100)
        ax.legend(fontsize=8)

    fig.suptitle("Average SOC Profile - G2V vs RB-V2G vs MPC-V2G (PV 2500 kW, SOC min 40%)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig("results/soc_profile_2500kW_min40soc.png", bbox_inches="tight")
    plt.close()
    print("  Saved: results/soc_profile_2500kW_min40soc.png")


# Figure 9: net power drawn from the grid (base + EV - PV)

def plot_net_grid_power(results: list, base_load_total_mw: np.ndarray, pv_sim: np.ndarray,
                        sim_start_h: int = 6):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    xlabels = _time_labels(sim_start_h)
    slots = np.arange(24)
    base_sim_kw = _base_load_sim(base_load_total_mw, sim_start_h)
    no_ev = base_sim_kw - pv_sim

    for ax, pen in zip(axes, [0.2, 0.5, 0.8]):
        ax.axhline(0, color="black", linewidth=0.8)
        ax.plot(slots, no_ev, color="grey", linewidth=1.5, linestyle=":", label="Base - PV (no EV)")

        stats = []
        for strat in STRATS:
            r = _find_result(results, pen, strat)
            if r is None:
                continue
            net = base_sim_kw + r["ev_kw"] - pv_sim
            ax.plot(slots, net, color=COLORS_STRAT[strat], linestyle=LINESTYLES[strat],
                    linewidth=2.5 if strat == "MPC-V2G" else 2, label=LABELS[strat])
            stats.append(f"{strat}: std {net.std():.0f} kW, peak {net.max():.0f} kW")

        ax.annotate("\n".join(stats), xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.set_title(f"EV penetration {int(pen * 100)}%")
        ax.set_xticks(slots[::3])
        ax.set_xticklabels(xlabels[::3], rotation=45, ha="right")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Net power from grid [kW]")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)

    fig.suptitle("Net Grid Power (Base + EV - PV) - G2V vs RB-V2G vs MPC-V2G (PV 2500 kW, SOC min 40%)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig("results/net_grid_power_2500kW_min40soc.png", bbox_inches="tight")
    plt.close()
    print("  Saved: results/net_grid_power_2500kW_min40soc.png")


# Master call

def plot_all(results: list, pv_sim: np.ndarray, base_load_total_mw: np.ndarray, sim_start_h: int = 6):
    print("\nPlotting figures:")
    plot_ev_load_curves(results, sim_start_h)
    plot_voltage_profiles(results, sim_start_h)
    plot_voltage_heatmaps(results, sim_start_h)
    plot_metrics_barchart(results)
    plot_pv_and_load(pv_sim, base_load_total_mw, sim_start_h)
    plot_total_load_curves(results, base_load_total_mw, sim_start_h, pv_sim)
    plot_ev_delta(results, sim_start_h)
    plot_soc_profile(results, sim_start_h)
    plot_net_grid_power(results, base_load_total_mw, pv_sim, sim_start_h)
    print("All figures saved to results/")
