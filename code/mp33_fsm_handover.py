#!/usr/bin/env python3
"""
Mobile Communications - Unit 3 - Mini Project 3.3
Device connection and handover modelled as a finite-state machine.

States : IDLE, RA_PREAMBLE, RA_RAR, RA_MSG3, RA_CONTENTION  (Random-Access = 4 sub-states, Msg1..Msg4)
         CONNECTED, HANDOVER
Failure: Radio Link Failure (RLF) -> IDLE  (too-late handover), handover failure -> IDLE
Policies are interchangeable objects: decide(t, serving, rsrp_dict) -> target cell or None.

Run: python3 mp33_fsm_handover.py  (logs -> ../output, figures -> ../figures)
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG, OUT = os.path.join(HERE, "..", "figures"), os.path.join(HERE, "..", "output")
os.makedirs(FIG, exist_ok=True); os.makedirs(OUT, exist_ok=True)
NAVY, BLUE, TEAL, ORANGE, RED, GREY, GREEN = "#1F3864", "#2E75B6", "#00A3A1", "#ED7D31", "#C00000", "#7F7F7F", "#70AD47"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": NAVY, "axes.titleweight": "bold",
                     "axes.titlecolor": NAVY, "axes.labelcolor": NAVY, "figure.dpi": 120})

# ------------------------------------------------------------- scenario parameters
DT = 0.25                    # s per sample
SPEED = 20.0                 # m/s (72 km/h)
SITES = {"A": 0.0, "B": 800.0, "C": 1600.0}      # cell positions along the road (m)
X0, X1 = -100.0, 1700.0
PL_EXP, PL_REF = 32.0, -22.0          # RSRP(d) = PL_REF - PL_EXP*log10(d)
SHADOW_SIGMA, SHADOW_CORR = 3.5, 0.9
RLF_THR, RLF_COUNT = -112.0, 3        # RLF if serving < thr for 3 consecutive samples
CAMP_THR = -118.0                     # minimum RSRP to attempt random access
HO_DURATION = 2                       # samples spent in HANDOVER (execution phase)
REESTABLISH = 4                       # samples spent searching after RLF
P_NO_RAR, P_COLLISION = 0.10, 0.15    # RACH impairments (initial access only)
MAX_PREAMBLE_TRIES = 4
PING_PONG_WINDOW = 8                  # samples (= 2 s)


def build_scenario(seed):
    """Scripted A -> B -> C drive with correlated log-normal shadowing: returns t, x, dict of RSRP series."""
    rng = np.random.default_rng(seed)
    n = int((X1 - X0) / SPEED / DT)
    x = X0 + SPEED * DT * np.arange(n)
    sig = {}
    for c, pos in SITES.items():
        d = np.maximum(np.abs(x - pos), 10.0)
        shadow = np.zeros(n)
        for k in range(1, n):
            shadow[k] = SHADOW_CORR * shadow[k - 1] + math.sqrt(1 - SHADOW_CORR ** 2) * rng.normal(0, SHADOW_SIGMA)
        fast = rng.normal(0, 1.5, n)              # measurement noise / fast fading residue
        sig[c] = PL_REF - PL_EXP * np.log10(d) + shadow + fast
    return np.arange(n), x, sig


# ------------------------------------------------------------- interchangeable policies
class NaivePolicy:
    name = "Naive"
    def reset(self): pass
    def decide(self, t, serving, rsrp):
        best = max((c for c in rsrp if c != serving), key=lambda c: rsrp[c])
        return best if rsrp[best] > rsrp[serving] else None


class HystTTTPolicy:
    def __init__(self, margin=3.0, ttt=5):
        self.margin, self.ttt = margin, ttt
        self.name = f"Hyst {margin:g} dB + TTT {ttt}"
        self.reset()
    def reset(self):
        self.cand, self.streak = None, 0
    def decide(self, t, serving, rsrp):
        best = max((c for c in rsrp if c != serving), key=lambda c: rsrp[c])
        if rsrp[best] > rsrp[serving] + self.margin:
            self.streak = self.streak + 1 if best == self.cand else 1
            self.cand = best
            if self.streak >= self.ttt:
                self.reset()
                return best
        else:
            self.reset()
        return None


# ------------------------------------------------------------- the state machine
class Device:
    RA_STEPS = ["RA_PREAMBLE", "RA_RAR", "RA_MSG3", "RA_CONTENTION"]

    def __init__(self, policy, rng):
        self.p, self.rng = policy, rng
        self.state, self.cell, self.target = "IDLE", None, None
        self.ra_idx, self.tries, self.ho_left, self.bad, self.wait = 0, 0, 0, 0, 0
        self.log, self.state_hist, self.cell_hist = [], [], []
        self.handovers, self.ho_fail, self.rlf = [], 0, 0

    def go(self, t, new, cell, trigger):
        self.log.append((t, self.state, new, cell, trigger))
        self.state = new

    def step(self, t, rsrp):
        s = self.state
        if s == "IDLE":
            if self.wait > 0:
                self.wait -= 1
            else:
                best = max(rsrp, key=rsrp.get)
                if rsrp[best] > CAMP_THR:
                    self.cell, self.tries, self.ra_idx = best, 1, 0
                    self.go(t, "RA_PREAMBLE", best, f"cell {best} selected ({rsrp[best]:.1f} dBm); Msg1 preamble sent")
        elif s == "RA_PREAMBLE":                       # Msg1 -> wait for Msg2
            if self.rng.random() < P_NO_RAR:
                if self.tries >= MAX_PREAMBLE_TRIES:
                    self.go(t, "IDLE", None, "max preamble attempts reached"); self.cell = None
                else:
                    self.tries += 1
                    self.log.append((t, s, s, self.cell, f"no RAR - power ramp, retry #{self.tries}"))
            else:
                self.go(t, "RA_RAR", self.cell, "Msg2 RAR received: timing advance + UL grant + temp-ID")
        elif s == "RA_RAR":
            self.go(t, "RA_MSG3", self.cell, "Msg3 RRC Connection Request sent with UE identity")
        elif s == "RA_MSG3":
            self.go(t, "RA_CONTENTION", self.cell, "waiting for Msg4 contention resolution")
        elif s == "RA_CONTENTION":
            if self.rng.random() < P_COLLISION:
                self.tries += 1
                self.go(t, "RA_PREAMBLE", self.cell, "contention lost (identity mismatch) - new preamble")
            else:
                self.bad = 0
                self.go(t, "CONNECTED", self.cell, "Msg4 echoes UE identity - contention resolved")
        elif s in ("CONNECTED", "HANDOVER"):
            self.bad = self.bad + 1 if rsrp[self.cell] < RLF_THR else 0
            if self.bad >= RLF_COUNT:                  # ---- failure path (too-late handover)
                self.rlf += 1
                self.go(t, "IDLE", None, f"RLF: serving {self.cell} {rsrp[self.cell]:.1f} dBm < {RLF_THR} for {RLF_COUNT} samples")
                self.cell, self.target, self.wait = None, None, REESTABLISH
                self.p.reset()
                return
            if s == "CONNECTED":
                tgt = self.p.decide(t, self.cell, rsrp)
                if tgt:
                    self.target, self.ho_left = tgt, HO_DURATION
                    self.go(t, "HANDOVER", self.cell,
                            f"{self.p.name}: {tgt} {rsrp[tgt]:.1f} vs {self.cell} {rsrp[self.cell]:.1f} dBm -> HO command")
            else:
                self.ho_left -= 1
                if self.ho_left <= 0:
                    if rsrp[self.target] < RLF_THR:     # target unusable -> handover failure
                        self.ho_fail += 1
                        self.go(t, "IDLE", None, f"HO failure: target {self.target} {rsrp[self.target]:.1f} dBm too weak")
                        self.cell, self.target, self.wait = None, None, REESTABLISH
                    else:
                        old = self.cell
                        self.handovers.append((t, old, self.target))
                        self.cell, self.bad = self.target, 0
                        self.go(t, "CONNECTED", self.cell,
                                f"contention-free RACH to {self.cell} done (dedicated preamble, TA applied)")
                        self.target = None

    def run(self, sig):
        n = len(next(iter(sig.values())))
        for t in range(n):
            r = {c: float(sig[c][t]) for c in sig}
            self.step(t, r)
            self.state_hist.append(self.state)
            self.cell_hist.append(self.cell)
        return self


def count_ping_pongs(handovers, window=PING_PONG_WINDOW):
    c = 0
    for a, b in zip(handovers, handovers[1:]):
        if b[2] == a[1] and b[0] - a[0] <= window:
            c += 1
    return c


def summarize(dev):
    n = len(dev.state_hist)
    conn = sum(1 for s in dev.state_hist if s in ("CONNECTED", "HANDOVER"))
    return dict(handovers=len(dev.handovers), ho_fail=dev.ho_fail, rlf=dev.rlf,
                ping_pong=count_ping_pongs(dev.handovers), connected_pct=100 * conn / n)


# ------------------------------------------------------------- figures
def draw_state_diagram(path):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.6); ax.axis("off")
    W, H = 1.9, 0.8
    boxes = {"IDLE": (1.4, 2.8, GREY, "IDLE"), "M1": (3.8, 5.3, BLUE, "RA Msg1\nPreamble"),
             "M2": (6.8, 5.3, BLUE, "RA Msg2\nRAR + TA"), "M3": (9.8, 5.3, BLUE, "RA Msg3\nConn. Request"),
             "M4": (9.8, 3.8, BLUE, "RA Msg4\nContention res."), "CONN": (6.8, 2.4, GREEN, "CONNECTED"),
             "HO": (6.8, 0.6, ORANGE, "HANDOVER\nin progress")}
    for x, y, c, t in boxes.values():
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.04", fc=c, ec=NAVY, lw=1.5))
        ax.text(x, y, t, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")

    def arrow(p1, p2, col=NAVY, rad=0.0):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15, color=col, lw=1.7,
                                     connectionstyle=f"arc3,rad={rad}"))

    def label(x, y, t, col=NAVY, ha="center"):
        ax.text(x, y, t, fontsize=8.3, color=col, ha=ha, va="center", bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5))

    arrow((1.4, 3.2), (2.85, 5.0)); label(1.3, 4.4, "cell found\nRSRP > -118 dBm")
    arrow((4.75, 5.3), (5.85, 5.3)); label(5.3, 4.85, "preamble\ndetected")
    arrow((7.75, 5.3), (8.85, 5.3)); label(8.3, 4.85, "TA + grant\n+ temp-ID")
    arrow((9.8, 4.9), (9.8, 4.2)); label(10.75, 4.55, "Msg3 sent")
    arrow((8.85, 3.7), (7.6, 2.8), GREEN); label(8.75, 3.0, "Msg4 ID match", GREEN)
    arrow((10.75, 3.9), (3.8, 5.7), RED, rad=0.32); label(9.4, 6.6, "collision / no RAR: retry preamble", RED)
    arrow((6.5, 2.0), (6.5, 1.0), ORANGE); label(5.0, 1.55, "policy fires\n(naive or hyst + TTT)", ORANGE)
    arrow((7.1, 1.0), (7.1, 2.0), GREEN); label(8.55, 1.55, "target RACH ok\n(contention-free)", GREEN)
    arrow((5.85, 2.55), (2.35, 2.85), RED); label(4.1, 3.25, "RLF: serving < -112 dBm x 3", RED)
    arrow((5.85, 0.6), (1.6, 2.4), RED, rad=-0.2); label(2.7, 0.9, "RLF / handover failure", RED)
    ax.text(0.2, 7.3, "Device connection & handover finite-state machine", fontsize=14, fontweight="bold", color=NAVY)
    ax.text(0.2, 6.95, "Blue = Random Access (4 logical steps)   Red = failure paths   Orange = handover", fontsize=9, color=GREY)
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


STATE_Y = {"IDLE": 0, "RA_PREAMBLE": 1, "RA_RAR": 1, "RA_MSG3": 1, "RA_CONTENTION": 1, "CONNECTED": 2, "HANDOVER": 3}


def draw_runs(t, x, sig, devs, path):
    fig, axes = plt.subplots(len(devs) + 1, 1, figsize=(11, 3 + 2.6 * len(devs)), sharex=True,
                             gridspec_kw={"height_ratios": [1.5] + [1.3] * len(devs)})
    cols = {"A": BLUE, "B": ORANGE, "C": TEAL}
    for c in sig:
        axes[0].plot(t * DT, sig[c], color=cols[c], lw=1.2, label=f"Cell {c}")
    axes[0].axhline(RLF_THR, color=RED, ls="--", lw=1); axes[0].text(1, RLF_THR + 1, "RLF threshold -112 dBm", color=RED, fontsize=8)
    axes[0].set_ylabel("RSRP (dBm)"); axes[0].legend(ncol=3, fontsize=8, loc="upper center")
    axes[0].set_title("Scripted A -> B -> C drive (72 km/h, 800 m site spacing)"); axes[0].grid(alpha=0.25)
    for ax, dev in zip(axes[1:], devs):
        sy = [STATE_Y[s] for s in dev.state_hist]
        ax.step(t * DT, sy, color=NAVY, lw=1.6, where="post")
        for (k, a, b) in dev.handovers:
            ax.axvline(k * DT, color=GREEN, lw=1, alpha=0.7)
        for (tt, f, to, cell, trig) in dev.log:
            if trig.startswith("RLF"):
                ax.axvline(tt * DT, color=RED, lw=1.6)
        ax.set_yticks([0, 1, 2, 3]); ax.set_yticklabels(["IDLE", "RACH", "CONN", "HO"], fontsize=8)
        s = summarize(dev)
        ax.set_title(f"{dev.p.name}: {s['handovers']} HO, {s['rlf']} RLF, {s['ping_pong']} ping-pong "
                     f"(green = handover done, red = RLF)", fontsize=10)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def write_log(dev, path, limit=None):
    with open(path, "w") as f:
        f.write(f"Transition log - policy: {dev.p.name}\n")
        f.write(f"{'t(s)':>6}  {'from':<14}{'to':<14}{'cell':<5} trigger\n" + "-" * 110 + "\n")
        for (t, a, b, c, trig) in (dev.log if not limit else dev.log[:limit]):
            f.write(f"{t * DT:>6.2f}  {a:<14}{b:<14}{str(c):<5} {trig}\n")


def main():
    policies = lambda: [NaivePolicy(), HystTTTPolicy(3.0, 5), HystTTTPolicy(10.0, 12)]
    draw_state_diagram(os.path.join(FIG, "fig11_fsm_diagram.png"))
    SEED = 2
    t, x, sig = build_scenario(SEED)
    devs = []
    for pol in policies():
        d = Device(pol, np.random.default_rng(SEED + 1)).run(sig)
        devs.append(d)
        write_log(d, os.path.join(OUT, f"mp33_log_{pol.name.replace(' ', '_').replace('+', 'plus')}.txt"))
    draw_runs(t, x, sig, devs, os.path.join(FIG, "fig12_scenario_runs.png"))
    # save generated series
    np.savetxt(os.path.join(OUT, "mp33_signal_series.csv"),
               np.column_stack([t * DT, x, sig["A"], sig["B"], sig["C"]]), delimiter=",",
               header="time_s,position_m,rsrp_A,rsrp_B,rsrp_C", comments="", fmt="%.3f")
    print(f"Scenario seed {SEED}: {len(t)} samples, {len(t) * DT:.0f} s")
    print(f"{'Policy':<24}{'HO':>4}{'HOfail':>8}{'RLF':>5}{'PingPong':>10}{'Conn%':>8}")
    single = []
    for d in devs:
        s = summarize(d); single.append((d.p.name, s))
        print(f"{d.p.name:<24}{s['handovers']:>4}{s['ho_fail']:>8}{s['rlf']:>5}{s['ping_pong']:>10}{s['connected_pct']:>8.1f}")

    # Monte-Carlo over 300 random scenarios for statistically sound comparison
    N = 300
    agg = {p.name: [] for p in policies()}
    for sd in range(1000, 1000 + N):
        tt, xx, ss = build_scenario(sd)
        for pol in policies():
            s = summarize(Device(pol, np.random.default_rng(sd + 7)).run(ss))
            agg[pol.name].append(s)
    print(f"\nMonte-Carlo ({N} scenarios) mean per run:")
    mc = []
    for k, v in agg.items():
        m = {key: float(np.mean([r[key] for r in v])) for key in v[0]}
        mc.append((k, m))
        print(f"{k:<24}{m['handovers']:>6.2f}{m['ho_fail']:>8.2f}{m['rlf']:>7.2f}{m['ping_pong']:>10.2f}{m['connected_pct']:>8.1f}")
    with open(os.path.join(OUT, "mp33_summary.txt"), "w") as f:
        f.write("Single run (seed 2)\n")
        for n_, s in single:
            f.write(f"{n_}: {s}\n")
        f.write(f"\nMonte-Carlo {N} runs\n")
        for n_, m in mc:
            f.write(f"{n_}: {m}\n")
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    names = [k for k, _ in mc]
    for a, key, ttl, col in zip(ax, ["handovers", "ping_pong", "rlf"], ["Handovers / run", "Ping-pongs / run", "RLFs / run"],
                                [BLUE, ORANGE, RED]):
        vals = [m[key] for _, m in mc]
        a.bar(["Naive", "3 dB\nTTT 5", "10 dB\nTTT 12"], vals, color=col)
        for i, v in enumerate(vals):
            a.text(i, v + 0.02 * max(vals + [1]), f"{v:.2f}", ha="center", color=NAVY, fontsize=9, fontweight="bold")
        a.set_title(ttl); a.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Monte-Carlo comparison over {N} random shadowing realisations", color=NAVY, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(os.path.join(FIG, "fig13_policy_comparison.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
