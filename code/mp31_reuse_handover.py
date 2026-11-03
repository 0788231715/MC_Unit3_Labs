#!/usr/bin/env python3
"""
Mobile Communications - Unit 3 - Mini Project 3.1
Simulating Frequency Reuse Planning and Handover Decision Logic

Part (a): hexagonal frequency-reuse pattern generator, Q = D/R = sqrt(3N)
Part (b): handover decision simulator (naive / hysteresis / hysteresis + TTT)
Lab Tasks 1-6 are implemented in the second half of this file.

Run:  python3 mp31_reuse_handover.py      (figures -> ../figures, tables -> ../output)
"""
import math
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Wedge
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
OUT = os.path.join(HERE, "..", "output")
os.makedirs(FIG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

NAVY, BLUE, TEAL, ORANGE, RED, GREY = "#1F3864", "#2E75B6", "#00A3A1", "#ED7D31", "#C00000", "#7F7F7F"
GROUP_COLORS = ["#1F3864", "#2E75B6", "#00A3A1", "#ED7D31", "#70AD47", "#7030A0",
                "#FFC000", "#C00000", "#8FAADC", "#A9D18E", "#F4B183", "#BFBFBF"]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": NAVY, "axes.titleweight": "bold",
                     "axes.titlecolor": NAVY, "axes.labelcolor": NAVY, "figure.dpi": 120})

# --------------------------------------------------------------------------
# Step 1: environment check
# --------------------------------------------------------------------------
print("NumPy", np.__version__, "| Matplotlib", matplotlib.__version__, "| Python", sys.version.split()[0])

# --------------------------------------------------------------------------
# PART (a) - hexagonal geometry and frequency-group assignment
# --------------------------------------------------------------------------
def hex_centers(rings, R=1.0):
    """Step 2: centres of hexagons in `rings` rings around the origin (pointy-top, axial coords).
    Returns list of (q, r, x, y)."""
    cells = []
    for q in range(-rings, rings + 1):
        for r in range(max(-rings, -q - rings), min(rings, -q + rings) + 1):
            x = R * math.sqrt(3) * (q + r / 2.0)
            y = R * 1.5 * r
            cells.append((q, r, x, y))
    return cells


def find_ij(N):
    """Integers i>=j>=0 with N = i^2 + i*j + j^2 (None if N cannot tile a hex grid)."""
    for i in range(0, 12):
        for j in range(0, i + 1):
            if i * i + i * j + j * j == N:
                return i, j
    return None


def group_labels(cells, N):
    """Step 3: frequency-group label 1..N for each cell using the cluster lattice
    spanned by u=(i, j) and v=(-j, i+j) (axial coordinates, determinant = N).
    Two cells share a label iff their difference is an integer combination of u and v."""
    ij = find_ij(N)
    if ij is None:
        raise ValueError(f"N={N} does not satisfy N=i^2+ij+j^2")
    i, j = ij
    rem = []
    for (q, r, _, _) in cells:
        s = ((i + j) * q + j * r) // N          # floor of lattice coordinate along u
        t = (-j * q + i * r) // N               # floor of lattice coordinate along v
        rem.append((q - s * i + t * j, r - s * j - t * (i + j)))   # remainder inside fundamental cell
    order = {k: n + 1 for n, k in enumerate(sorted(set(rem)))}      # stable label numbering
    return [order[k] for k in rem]


def min_cochannel_distance(cells, labels, R=1.0):
    """Measured nearest same-group distance D (in units of R) - used to verify Q = sqrt(3N)."""
    best = {}
    xy = np.array([[c[2], c[3]] for c in cells])
    lab = np.array(labels)
    for g in set(labels):
        pts = xy[lab == g]
        if len(pts) < 2:
            continue
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        d[d == 0] = 1e9
        best[g] = d.min() / R
    return min(best.values()) if best else float("nan")


def draw_layout(ax, N, rings, R=1.0, sectors=False, title=None):
    cells = hex_centers(rings, R)
    labels = group_labels(cells, N)
    for (q, r, x, y), g in zip(cells, labels):
        ax.add_patch(RegularPolygon((x, y), 6, radius=R, orientation=0,
                                    facecolor=GROUP_COLORS[(g - 1) % len(GROUP_COLORS)],
                                    edgecolor="white", linewidth=1.2, alpha=0.95))
        txtc = "white" if g in (1, 2, 3, 6, 8) else NAVY
        ax.text(x, y, str(g), ha="center", va="center", fontsize=7 if len(cells) > 60 else 9,
                color=txtc, fontweight="bold")
        if sectors:   # Lab task 6: three 120-degree sectors drawn as spokes
            for ang in (90, 210, 330):
                ax.plot([x, x + 0.9 * R * math.cos(math.radians(ang))],
                        [y, y + 0.9 * R * math.sin(math.radians(ang))], color="white", lw=0.8, alpha=0.8)
    lim = (rings + 1.2) * R * math.sqrt(3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title or f"N = {N}   (Q = {math.sqrt(3 * N):.2f})")
    return cells, labels


def part_a(Ns=(1, 3, 4, 7), rings=5, fname="fig1_reuse_layouts.png", extra=False):
    rows = []
    cols = 2
    nrows = math.ceil(len(Ns) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(11, 5.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, N in zip(axes, Ns):
        cells, labels = draw_layout(ax, N, rings)
        Q = math.sqrt(3 * N)
        Dm = min_cochannel_distance(cells, labels)
        ij = find_ij(N)
        rows.append((N, ij, Q, len(set(labels)), Dm))
    for ax in axes[len(Ns):]:
        ax.axis("off")
    fig.suptitle("Frequency-reuse layouts: hexagon colour & number = frequency group",
                 color=NAVY, fontweight="bold", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG, fname), dpi=140)
    plt.close(fig)
    return rows


def print_q_table(rows, path):
    lines = ["N    (i,j)   Q=sqrt(3N)   groups_in_layout   measured_D/R"]
    for N, ij, Q, g, Dm in rows:
        lines.append(f"{N:<4d} {str(ij):<7s} {Q:<12.3f} {g:<18d} {Dm:.3f}")
    txt = "\n".join(lines)
    print(txt)
    with open(path, "w") as f:
        f.write(txt + "\n")


# --------------------------------------------------------------------------
# PART (b) - synthetic RSRP series and handover triggers
# --------------------------------------------------------------------------
def make_series(n=300, start=-76.0, end=-108.0, walk_sigma=0.35, noise_sigma=2.6, rng=None):
    """Steps 7-8: linear trend + random walk + white Gaussian noise (dBm)."""
    rng = rng or np.random.default_rng(7)
    t = np.arange(n)
    trend = start + (end - start) * t / (n - 1)
    walk = np.cumsum(rng.normal(0, walk_sigma, n))
    walk -= np.linspace(walk[0], walk[-1], n)        # bridge the walk so the endpoints stay on-trend
    return trend + walk + rng.normal(0, noise_sigma, n)


def run_trigger(series, serving0=0, margin=0.0, ttt=1, bias=None):
    """Generic handover engine for any number of cells.
    margin : hysteresis margin in dB (0 = naive)
    ttt    : consecutive samples the entering condition must hold (1 = no timer)
    bias   : optional dict {cell: dB offset} added to a candidate (load balancing, Lab 5)
    Returns serving-index array and list of (time, from, to)."""
    n_cells, n = series.shape
    serving = serving0
    streak, cand = 0, None
    serv_hist, events = np.zeros(n, dtype=int), []
    for k in range(n):
        vals = series[:, k].copy()
        if bias:
            for c, b in bias.items():
                vals[c] += b
        others = [c for c in range(n_cells) if c != serving]
        best = max(others, key=lambda c: vals[c])
        cond = vals[best] > series[serving, k] + margin
        if cond:
            streak = streak + 1 if best == cand else 1
            cand = best
            if streak >= ttt:
                events.append((k, serving, best))
                serving, streak, cand = best, 0, None
        else:
            streak, cand = 0, None
        serv_hist[k] = serving
    return serv_hist, events


def serving_rsrp(series, serv_hist):
    return series[serv_hist, np.arange(series.shape[1])]


def ping_pongs(events, window=10):
    """Count handovers that return to the cell left within `window` samples."""
    c = 0
    for a, b in zip(events, events[1:]):
        if b[2] == a[1] and (b[0] - a[0]) <= window:
            c += 1
    return c


def part_b():
    rng = np.random.default_rng(7)
    A = make_series(rng=rng, start=-76, end=-108)       # serving: device moving away
    B = make_series(rng=rng, start=-108, end=-76)       # neighbour: device approaching
    S = np.vstack([A, B])
    t = np.arange(S.shape[1])
    configs = [("Naive (no margin, no timer)", 0.0, 1),
               ("Hysteresis only (3 dB)", 3.0, 1),
               ("Hysteresis 3 dB + TTT 5 samples", 3.0, 5)]
    THR = -100.0
    results = []
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True)
    for ax, (name, m, ttt) in zip(axes, configs):
        sh, ev = run_trigger(S, 0, m, ttt)
        srv = serving_rsrp(S, sh)
        ax.plot(t, S[0], color=BLUE, lw=1.0, alpha=0.85, label="Cell A RSRP")
        ax.plot(t, S[1], color=ORANGE, lw=1.0, alpha=0.85, label="Cell B RSRP")
        ax.plot(t, srv, color=NAVY, lw=1.5, alpha=0.9, drawstyle="steps-post", label="Serving-cell RSRP")
        for (k, a, b) in ev:
            ax.axvline(k, color=RED, lw=0.9, alpha=0.7)
        ax.axhline(THR, color=GREY, ls="--", lw=1)
        ax.text(2, THR + 0.6, "minimum threshold -100 dBm", fontsize=8, color=GREY)
        ax.set_ylabel("RSRP (dBm)")
        ax.set_title(f"{name}: {len(ev)} handovers")
        ax.grid(alpha=0.25)
        below = int(np.sum(srv < THR))
        results.append((name, len(ev), ping_pongs(ev), below, float(srv.min())))
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Sample index (time)")
    fig.suptitle("Handover behaviour on identical RSRP traces (red lines = handovers)",
                 color=NAVY, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(FIG, "fig2_handover_three_configs.png"), dpi=140)
    plt.close(fig)

    # raw curves figure (step 10 style) with naive markers
    fig, ax = plt.subplots(figsize=(11, 4.2))
    sh, ev = run_trigger(S, 0, 0, 1)
    ax.plot(t, S[0], color=BLUE, label="Cell A"); ax.plot(t, S[1], color=ORANGE, label="Cell B")
    for (k, a, b) in ev:
        ax.axvline(k, color=RED, alpha=0.5, lw=0.9)
    ax.set_title(f"Naive trigger on raw series: {len(ev)} handovers clustered at the crossover")
    ax.set_xlabel("Sample index"); ax.set_ylabel("RSRP (dBm)"); ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_naive_pingpong.png"), dpi=140); plt.close(fig)

    lines = ["Configuration                              Handovers  Ping-pongs  Samples<-100dBm  MinServingRSRP(dBm)"]
    for (n_, h, pp, b_, mn) in results:
        lines.append(f"{n_:<42s} {h:<10d} {pp:<11d} {b_:<16d} {mn:.1f}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "mp31_handover_counts.txt"), "w").write(txt + "\n")
    return S, results


# --------------------------------------------------------------------------
# LAB TASKS
# --------------------------------------------------------------------------
def lab_task1():
    """Extend generator to N = 9 and 12."""
    rows = part_a(Ns=(7, 9, 12, 4), rings=6, fname="fig4_lab1_N9_N12.png")
    print("\n[Lab task 1] N=9 and N=12 added")
    print_q_table(rows, os.path.join(OUT, "mp31_lab1_q_table.txt"))


def lab_task2(S2):
    """Third neighbour: strongest of three cells at each decision point."""
    rng = np.random.default_rng(21)
    C = make_series(rng=rng, start=-112, end=-84)     # a third cell that approaches later than B
    S3 = np.vstack([S2, C])
    out = []
    for name, m, ttt in [("Naive", 0, 1), ("Hyst 3 dB", 3, 1), ("Hyst 3 dB + TTT 5", 3, 5)]:
        sh, ev = run_trigger(S3, 0, m, ttt)
        out.append((name, len(ev), ping_pongs(ev)))
    print("\n[Lab task 2] strongest-of-three:", out)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    t = np.arange(S3.shape[1])
    for c, col, lab in zip(range(3), (BLUE, ORANGE, TEAL), "ABC"):
        ax.plot(t, S3[c], color=col, lw=0.9, alpha=0.8, label=f"Cell {lab}")
    sh, ev = run_trigger(S3, 0, 3, 5)
    ax.plot(t, serving_rsrp(S3, sh), color=NAVY, lw=2.4, drawstyle="steps-post", label="Serving (3 dB + TTT 5)")
    ax.set_title("Lab 2: three-cell trigger always selects the strongest candidate")
    ax.set_xlabel("Sample index"); ax.set_ylabel("RSRP (dBm)"); ax.legend(ncol=4, fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_lab2_three_cells.png"), dpi=140); plt.close(fig)
    return out


def lab_task3(S):
    """Sweep margin x TTT -> handover count and minimum serving RSRP (MRO-style optimisation)."""
    margins = np.arange(0, 11, 1.0)
    ttts = np.arange(1, 21, 1)
    # average over several noise realisations for a stable optimum
    reps = 12
    cnt = np.zeros((len(ttts), len(margins)))
    mn = np.zeros_like(cnt)
    for rep in range(reps):
        rg = np.random.default_rng(100 + rep)
        SS = np.vstack([make_series(rng=rg, start=-76, end=-108), make_series(rng=rg, start=-108, end=-76)])
        for a, ttt in enumerate(ttts):
            for b, m in enumerate(margins):
                sh, ev = run_trigger(SS, 0, m, int(ttt))
                cnt[a, b] += len(ev) / reps
                mn[a, b] += serving_rsrp(SS, sh).min() / reps
    THR = -100.0
    feasible = mn >= THR
    masked = np.where(feasible, cnt, np.inf)
    cmin = masked.min()
    # among near-minimal counts, prefer the combination that keeps serving RSRP highest (lowest RLF risk)
    cand = np.where(masked <= cmin + 0.25, mn, -np.inf)
    a, b = np.unravel_index(np.argmax(cand), cand.shape)
    best = (margins[b], ttts[a], cnt[a, b], mn[a, b])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    im0 = axes[0].imshow(cnt, origin="lower", aspect="auto", cmap="viridis_r",
                         extent=[margins[0] - .5, margins[-1] + .5, ttts[0] - .5, ttts[-1] + .5])
    axes[0].set_title("Mean handover count"); plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(mn, origin="lower", aspect="auto", cmap="RdYlGn",
                         extent=[margins[0] - .5, margins[-1] + .5, ttts[0] - .5, ttts[-1] + .5])
    axes[1].contour(margins, ttts, mn, levels=[THR], colors=[NAVY], linewidths=2)
    axes[1].set_title("Min serving RSRP (dBm), contour = -100 dBm"); plt.colorbar(im1, ax=axes[1])
    for ax in axes:
        ax.scatter([best[0]], [best[1]], marker="*", s=220, color=ORANGE, edgecolor="white", zorder=5)
        ax.set_xlabel("Hysteresis margin (dB)"); ax.set_ylabel("TTT (samples)")
    fig.suptitle(f"Lab 3: MRO-style sweep - best feasible: margin {best[0]:.0f} dB, TTT {best[1]} "
                 f"(mean {best[2]:.1f} handovers, min RSRP {best[3]:.1f} dBm)", color=NAVY, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, "fig6_lab3_sweep.png"), dpi=140); plt.close(fig)
    print("\n[Lab task 3] best feasible (margin, TTT, mean handovers, mean min RSRP):", tuple(round(float(x),2) for x in best))
    return best


def lab_task4(best):
    """Fast device: steeper trend + larger random walk; re-use tuned parameters."""
    m, ttt = best[0], int(best[1])
    res = {"slow": [], "fast": []}
    for rep in range(30):
        rg = np.random.default_rng(300 + rep)
        slow = np.vstack([make_series(rng=rg, start=-76, end=-108), make_series(rng=rg, start=-108, end=-76)])
        rg2 = np.random.default_rng(900 + rep)
        fastS = np.vstack([make_series(n=300, start=-66, end=-120, walk_sigma=0.9, rng=rg2),
                           make_series(n=300, start=-120, end=-66, walk_sigma=0.9, rng=rg2)])
        for key, SS in (("slow", slow), ("fast", fastS)):
            sh, ev = run_trigger(SS, 0, m, ttt)
            res[key].append((len(ev), serving_rsrp(SS, sh).min(), int(np.sum(serving_rsrp(SS, sh) < -100))))
    summ = {k: tuple(np.mean(v, axis=0)) for k, v in res.items()}
    print(f"\n[Lab task 4] tuned (m={m}, TTT={ttt}) slow vs fast (handovers, minRSRP, samples<-100):", summ)
    return summ, (m, ttt)


def lab_task5(S):
    """Asymmetric offset favouring handover toward cell B (biased threshold, load balancing)."""
    out = []
    for bias in (0.0, 2.0, 4.0, 6.0):
        sh, ev = run_trigger(S, 0, 3.0, 5, bias={1: bias})
        first = ev[0][0] if ev else None
        out.append((bias, len(ev), first))
    print("\n[Lab task 5] bias toward B (dB, handovers, first handover time):", out)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar([f"+{b:.0f} dB" for b, _, _ in out], [f if f is not None else 0 for _, _, f in out], color=TEAL)
    for i, (b, n_, f) in enumerate(out):
        ax.text(i, (f or 0) + 3, f"{n_} HO", ha="center", color=NAVY, fontweight="bold")
    ax.set_ylabel("Sample of first handover to B"); ax.set_xlabel("Bias added to Cell B")
    ax.set_title("Lab 5: biased threshold pulls the handover earlier"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_lab5_bias.png"), dpi=140); plt.close(fig)
    return out


def lab_task6():
    """Sectored hex layout (3 x 120 degrees) for N = 3 and N = 7 and interferer count comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, N in zip(axes, (3, 7)):
        draw_layout(ax, N, 3, sectors=True, title=f"N = {N}, 3-sector sites (spokes at 90/210/330 deg)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig8_lab6_sectored.png"), dpi=140); plt.close(fig)
    # first-tier co-channel interferers seen by a desired mobile: omni = 6, 120-degree sector ~ 2
    print("\n[Lab task 6] first-tier interferers: omni = 6, 120-deg sectors ~ 2 (see report)")
    return {"omni": 6, "sector120": 2, "sector60": 1}


# --------------------------------------------------------------------------
def main():
    rows = part_a()
    print("\nFrequency reuse summary (Step 6)")
    print_q_table(rows, os.path.join(OUT, "mp31_q_table.txt"))
    S, results = part_b()
    lab_task1()
    lab_task2(S)
    best = lab_task3(S)
    lab_task4(best)
    lab_task5(S)
    lab_task6()


if __name__ == "__main__":
    main()
