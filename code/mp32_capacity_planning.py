#!/usr/bin/env python3
"""
Mobile Communications - Unit 3 - Mini Project 3.2
Cellular Capacity and Frequency-Reuse Planning Tool

Computes channels/cell, number of cells, system capacity, Q and SIR for several reuse
factors N, then quantifies cell splitting and sectoring as alternative capacity levers.

Run: python3 mp32_capacity_planning.py   (tables -> ../output, charts -> ../figures)
"""
import csv
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG, OUT = os.path.join(HERE, "..", "figures"), os.path.join(HERE, "..", "output")
os.makedirs(FIG, exist_ok=True); os.makedirs(OUT, exist_ok=True)
NAVY, BLUE, TEAL, ORANGE, RED, GREY = "#1F3864", "#2E75B6", "#00A3A1", "#ED7D31", "#C00000", "#7F7F7F"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": NAVY, "axes.titleweight": "bold",
                     "axes.titlecolor": NAVY, "axes.labelcolor": NAVY, "figure.dpi": 120})

# ------------------------------------------------------------------ 1. CONFIGURATION
CONFIG = dict(
    total_spectrum_hz=25e6,        # 25 MHz allocation (one direction)
    channel_bw_hz=200e3,           # GSM-style 200 kHz carrier
    slots_per_channel=8,           # TDMA slots -> simultaneous users per channel (GSM)
    coverage_area_km2=1000.0,      # service area
    cell_radius_km=5.0,            # R
    candidate_N=[1, 3, 4, 7, 9, 12],
    path_loss_exp=4.0,             # urban macro propagation exponent n
)
HEX_FACTOR = 3 * math.sqrt(3) / 2   # hexagon area = 2.598 R^2


def hex_area(R):
    return HEX_FACTOR * R * R


def plan(cfg, R=None, N_list=None, sectors=1):
    """Return list of dict rows: N, channels per cell, cells, capacity, Q, SIR."""
    R = R or cfg["cell_radius_km"]
    total_ch = int(cfg["total_spectrum_hz"] // cfg["channel_bw_hz"])
    cells = cfg["coverage_area_km2"] / hex_area(R)
    i0 = {1: 6, 3: 2, 6: 1}[sectors]       # first-tier co-channel interferers (omni / 120 / 60 deg)
    rows = []
    for N in (N_list or cfg["candidate_N"]):
        ch_cell = total_ch // N
        Q = math.sqrt(3 * N)
        sir = Q ** cfg["path_loss_exp"] / i0
        rows.append(dict(N=N, total_channels=total_ch, channels_per_cell=ch_cell, cells=cells,
                         capacity_channels=ch_cell * cells,
                         capacity_users=ch_cell * cells * cfg["slots_per_channel"],
                         Q=Q, SIR_dB=10 * math.log10(sir), R=R, sectors=sectors))
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


def main():
    cfg = CONFIG
    print("CONFIGURATION:", cfg)
    base = plan(cfg)
    total_ch = base[0]["total_channels"]
    print(f"\nTotal channels = {cfg['total_spectrum_hz']/1e6:.0f} MHz / {cfg['channel_bw_hz']/1e3:.0f} kHz = {total_ch}")
    print(f"Hex cell area (R={cfg['cell_radius_km']} km) = {hex_area(cfg['cell_radius_km']):.2f} km2 "
          f"-> {base[0]['cells']:.1f} cells fit in {cfg['coverage_area_km2']:.0f} km2")
    hdr = f"{'N':>3} {'ch/cell':>8} {'cells':>7} {'capacity(ch)':>13} {'users(x8)':>10} {'Q':>6} {'SIR dB':>7}"
    print("\n" + hdr)
    for r in base:
        print(f"{r['N']:>3} {r['channels_per_cell']:>8} {r['cells']:>7.1f} {r['capacity_channels']:>13.0f} "
              f"{r['capacity_users']:>10.0f} {r['Q']:>6.2f} {r['SIR_dB']:>7.1f}")
    write_csv(base, os.path.join(OUT, "mp32_capacity_table.csv"))

    # ---- cell splitting: halve and quarter R (frequency plan N unchanged) ----
    N_split = 7
    split_rows = []
    for factor in (1, 2, 4):
        R = cfg["cell_radius_km"] / factor
        r = plan(cfg, R=R, N_list=[N_split])[0]
        # transmit power must fall by 10*n*log10(factor) dB to keep the same SIR/cell edge level
        r["power_reduction_dB"] = 10 * cfg["path_loss_exp"] * math.log10(factor)
        r["gain_vs_base"] = r["capacity_channels"] / base[[b["N"] for b in base].index(N_split)]["capacity_channels"]
        split_rows.append(r)
    print(f"\nCell splitting (N={N_split}):")
    for r in split_rows:
        print(f"  R={r['R']:.2f} km  cells={r['cells']:.0f}  capacity={r['capacity_channels']:.0f} ch  "
              f"gain x{r['gain_vs_base']:.1f}  Tx power change -{r['power_reduction_dB']:.0f} dB")
    write_csv(split_rows, os.path.join(OUT, "mp32_cell_splitting.csv"))

    # ---- sectoring: omni vs 120-deg sectors with tighter reuse ----
    sector_rows = []
    for label, N, sec in [("Omni, N=7", 7, 1), ("Omni, N=4", 4, 1), ("3-sector, N=4", 4, 3),
                          ("3-sector, N=3", 3, 3), ("6-sector, N=3", 3, 6)]:
        r = plan(cfg, N_list=[N], sectors=sec)[0]
        r["label"] = label
        sector_rows.append(r)
    print("\nSectoring (SIR = Q^n / i0, n=4):")
    for r in sector_rows:
        print(f"  {r['label']:<14} ch/cell={r['channels_per_cell']:>3}  Q={r['Q']:.2f}  SIR={r['SIR_dB']:.1f} dB  "
              f"capacity={r['capacity_channels']:.0f}")
    write_csv([{k: v for k, v in r.items() if k != 'label'} | {"label": r["label"]} for r in sector_rows],
              os.path.join(OUT, "mp32_sectoring.csv"))

    # ---------------------------------------------------------------- charts
    Ns = [r["N"] for r in base]
    cap = [r["capacity_channels"] for r in base]
    Qs = [r["Q"] for r in base]
    sirs = [r["SIR_dB"] for r in base]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    bars = ax[0].bar([str(n) for n in Ns], cap, color=[BLUE if n != 7 else ORANGE for n in Ns])
    for b, c in zip(bars, cap):
        ax[0].text(b.get_x() + b.get_width() / 2, c + 8, f"{c:.0f}", ha="center", color=NAVY, fontsize=9, fontweight="bold")
    ax[0].set_xlabel("Reuse factor N"); ax[0].set_ylabel("System capacity (channels)")
    ax[0].set_title("Capacity vs reuse factor"); ax[0].grid(axis="y", alpha=0.25)
    ax[1].plot(Qs, cap, "-o", color=NAVY, lw=2, label="Capacity (channels)")
    for q, c, n in zip(Qs, cap, Ns):
        ax[1].annotate(f"N={n}", (q, c), textcoords="offset points", xytext=(5, 6), fontsize=8)
    ax[1].set_xlabel("Co-channel reuse ratio Q = sqrt(3N)"); ax[1].set_ylabel("System capacity (channels)", color=NAVY)
    ax2 = ax[1].twinx()
    ax2.plot(Qs, sirs, "-s", color=ORANGE, lw=2, label="SIR (dB)")
    ax2.set_ylabel("SIR (dB), omni, n=4", color=ORANGE)
    ax[1].set_title("Capacity falls, SIR rises as Q grows"); ax[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig9_capacity_vs_N_Q.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    labels = ["Omni N=7\n(baseline)", "Omni N=4", "3-sector N=4", "3-sector N=3",
              "Split R/2\n(N=7)", "Split R/4\n(N=7)"]
    base7 = base[Ns.index(7)]["capacity_channels"]
    vals = [sector_rows[0]["capacity_channels"], sector_rows[1]["capacity_channels"],
            sector_rows[2]["capacity_channels"], sector_rows[3]["capacity_channels"],
            split_rows[1]["capacity_channels"], split_rows[2]["capacity_channels"]]
    sir_l = [sector_rows[0]["SIR_dB"], sector_rows[1]["SIR_dB"], sector_rows[2]["SIR_dB"],
             sector_rows[3]["SIR_dB"], split_rows[1]["SIR_dB"], split_rows[2]["SIR_dB"]]
    cols = [GREY, BLUE, TEAL, TEAL, ORANGE, ORANGE]
    bs = ax.bar(labels, vals, color=cols)
    for b, v, s_ in zip(bs, vals, sir_l):
        ax.text(b.get_x() + b.get_width() / 2, v + 20, f"{v:.0f}\nSIR {s_:.1f} dB", ha="center", fontsize=8.5, color=NAVY)
    ax.set_ylabel("System capacity (channels)"); ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("Capacity levers compared (25 MHz, 200 kHz channels, 1000 km2)"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig10_capacity_levers.png"), dpi=140); plt.close(fig)
    print("\nCharts written to ../figures")


if __name__ == "__main__":
    main()
