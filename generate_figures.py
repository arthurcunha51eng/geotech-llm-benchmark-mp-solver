"""
Generate all repository figures from solver results and LLM benchmark data.
Outputs PNG files to ./figures/
Run: python -X utf8 generate_figures.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
import os

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ─── Reference solver output (python -X utf8 main.py) ────────────────────────
REF_FS   = 1.9712681
REF_LAM  = 0.2265151
REF_ITERS = 7

ALPHA_DEG = np.array([39.691, 24.795, 11.543, -1.098, -13.795])
ALPHA     = np.radians(ALPHA_DEG)
X_MID     = np.array([23.471, 48.030, 72.588, 97.147, 121.705])
H         = np.array([8.816, 18.167, 20.088, 16.193, 6.843])
W         = np.array([3464.070, 7138.684, 7893.197, 6362.722, 2688.963])
E_REF     = np.array([0.0, 1517.944, 3077.714, 3070.487, 1541.944, 0.0])
X_SHEAR   = np.array([0.0, 343.837, 697.149, 695.512, 349.274, 0.0])

X_ENTRY = 11.192
X_EXIT  = 133.985
B       = (X_EXIT - X_ENTRY) / 5
X_BOUNDS = np.array([X_ENTRY + i * B for i in range(6)])

# Reference Ψ values (mixed-index, Zhu eq. 3.54) — analytically computed
PSI_REF = np.array([1.0421, 0.9796, 0.9306, 0.8767])

# Convergence logs: [F_iter_1, ..., F_converged] per model
REF_CONV_F   = [1.9274, 1.9767, 1.9706, 1.9713, 1.9713, 1.9713, 1.9713]
REF_CONV_LAM = [0.2293, 0.2262, 0.2266, 0.2265, 0.2265, 0.2265, 0.2265]

GEM_V0_F   = [1.9274, 1.9582, 1.9634, 1.9647, 1.9650]
GEM_V0_LAM = [0.1050, 0.1740, 0.2015, 0.2111, 0.2124]

GEM_V1_F   = [1.8964, 1.8850, 1.8812, 1.8810, 1.8810, 1.8810]
GEM_V1_LAM = [0.0000, 0.0450, 0.0521, 0.0531, 0.0531, 0.0531]

DS_V0_F    = [1.8262, 1.8348, 1.8517, 1.8736, 1.8879, 1.8857, 1.8857]
DS_V0_LAM  = [0.0000,-0.0500,-0.1200,-0.1500,-0.1420,-0.1418,-0.1418]

DS_V1_F    = [1.0000, 1.5200, 1.5410, 1.5419, 1.5419, 1.5419, 1.5419]
DS_V1_LAM  = [0.0000, 0.2000, 0.2310, 0.2335, 0.2336, 0.2337, 0.2337]

# ─── Palette ─────────────────────────────────────────────────────────────────
C_REF  = "#1B5E20"   # dark green — reference correct
C_GV0  = "#E65100"   # orange     — Gemini V0
C_GV1  = "#BF360C"   # deep orange — Gemini V1
C_DV0  = "#1A237E"   # navy       — DeepSeek V0
C_DV1  = "#6A1B9A"   # purple     — DeepSeek V1
C_MISS = "#B71C1C"   # red        — critical error

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Slope geometry, slip circle, slices
# ─────────────────────────────────────────────────────────────────────────────
def fig_slope_geometry():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_aspect("equal")

    # slope body polygon
    xs = [0, 0, 10, 134, 144, 144, 0]
    ys = [0, 46, 46,  15,  15,   0, 0]
    ax.fill(xs, ys, color="#E3F2FD", zorder=1)
    ax.plot(xs, ys, color="#37474F", lw=1.4, zorder=2)

    # slip circle arc
    theta = np.linspace(0, 2 * np.pi, 1000)
    xc, yc, R = 95.0, 120.0, 112.0
    arc_x = xc + R * np.cos(theta)
    arc_y = yc + R * np.sin(theta)
    mask = arc_y < np.interp(arc_x, [0, 10, 134, 144], [46, 46, 15, 15])
    ax.plot(arc_x[~mask], arc_y[~mask], "--", color="#546E7A",
            lw=1.3, alpha=0.55, zorder=3)

    # visible arc portion (below surface, inside slope body)
    x_arc = np.linspace(X_ENTRY, X_EXIT, 400)
    y_arc_bot = yc - np.sqrt(np.maximum(R**2 - (x_arc - xc)**2, 0))
    ax.plot(x_arc, y_arc_bot, color="#0277BD", lw=2.0, zorder=4, label="Slip circle")

    # slice fills + boundaries
    def y_top_fn(x):
        if x <= 10:   return 46.0
        elif x <= 134: return 46.0 - (46.0 - 15.0) / (134.0 - 10.0) * (x - 10.0)
        else:          return 15.0

    colors_active  = "#90CAF9"
    colors_passive = "#A5D6A7"
    for i in range(5):
        xl, xr = X_BOUNDS[i], X_BOUNDS[i + 1]
        xm = X_MID[i]
        # fill slice body
        xx = np.linspace(xl, xr, 30)
        yt = np.array([y_top_fn(x) for x in xx])
        yb = yc - np.sqrt(np.maximum(R**2 - (xx - xc)**2, 0))
        fc = colors_active if ALPHA_DEG[i] > 0 else colors_passive
        ax.fill_between(xx, yb, yt, color=fc, alpha=0.65, zorder=3)
        # slice boundaries (vertical)
        ax.axvline(xl, ymin=0, ymax=1, color="#37474F", lw=0.8,
                   ls="--", alpha=0.6, zorder=5)
        # slice label
        y_mid_slice = (y_top_fn(xm) + (yc - np.sqrt(R**2 - (xm - xc)**2))) / 2
        ax.text(xm, y_mid_slice + 1.5, f"{i+1}", ha="center",
                fontsize=8.5, color="#37474F", fontweight="bold", zorder=6)
        # alpha arrow label on slices 1 and 5
        if i == 0:
            ax.annotate(f"α₁ = {ALPHA_DEG[0]:.1f}°", xy=(xm, y_top_fn(xm) - 4),
                        fontsize=8, color="#0277BD", ha="center")
        if i == 4:
            ax.annotate(f"α₅ = {ALPHA_DEG[4]:.1f}°", xy=(xm, y_top_fn(xm) - 4),
                        fontsize=8, color="#388E3C", ha="center")

    ax.axvline(X_BOUNDS[-1], color="#37474F", lw=0.8, ls="--", alpha=0.6, zorder=5)

    # circle center
    ax.plot(xc, yc, "x", color="#B71C1C", ms=8, mew=2, zorder=6)
    ax.annotate(f"  Centre\n  ({xc}, {yc})", xy=(xc, yc),
                fontsize=8, color="#B71C1C", va="center")

    # zone labels
    ax.text(50, 38, "Active zone\n(α > 0, drives failure)",
            color="#1565C0", fontsize=8.5, ha="center",
            bbox=dict(fc="white", ec="none", alpha=0.7))
    ax.text(105, 22, "Passive zone\n(α < 0, resists)",
            color="#2E7D32", fontsize=8.5, ha="center",
            bbox=dict(fc="white", ec="none", alpha=0.7))

    # legend
    slip_line = mlines.Line2D([], [], color="#0277BD", lw=2, label="Slip arc")
    active_p  = mpatches.Patch(color=colors_active,  alpha=0.65, label="Active slices (1–3)")
    passive_p = mpatches.Patch(color=colors_passive, alpha=0.65, label="Passive slices (4–5)")
    ax.legend(handles=[slip_line, active_p, passive_p],
              loc="upper right", framealpha=0.9, fontsize=8.5)

    ax.set_xlabel("x  (m)", fontsize=10)
    ax.set_ylabel("y  (m)", fontsize=10)
    ax.set_title("Slope Geometry — Slip Circle and 5-Slice Discretization",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-5, 152)
    ax.set_ylim(-3, 55)
    ax.annotate(f"FS = {REF_FS:.4f}  |  λ = {REF_LAM:.4f}  |  n = 5  |  "
                f"c′ = 12.5 kPa  |  φ′ = 20°  |  γ = 16 kN/m³",
                xy=(0.5, -0.12), xycoords="axes fraction",
                ha="center", fontsize=8, color="#546E7A")
    fig.tight_layout()
    path = os.path.join(OUT, "fig1_slope_geometry.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — LLM Benchmark Scorecard (FS and λ)
# ─────────────────────────────────────────────────────────────────────────────
def fig_llm_benchmark():
    models = ["Reference\nsolver", "Gemini\nV0", "Gemini\nV1",
              "DeepSeek\nV0", "DeepSeek\nV1"]
    fs_vals  = [REF_FS, 1.965, 1.881, 1.886, 1.542]
    lam_vals = [REF_LAM, 0.212, 0.053, -0.142, 0.234]
    colors_fs  = [C_REF, C_GV0, C_GV1, C_DV0, C_DV1]
    colors_lam = [C_REF, C_GV0, C_GV1, C_DV0, C_DV1]

    x = np.arange(len(models))
    w = 0.55

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # FS bars
    bars1 = ax1.bar(x, fs_vals, width=w, color=colors_fs, zorder=3,
                    edgecolor="white", linewidth=0.8)
    ax1.axhline(REF_FS, color=C_REF, lw=1.5, ls="--", zorder=4,
                label=f"Reference FS = {REF_FS:.4f}")
    ax1.axhspan(REF_FS * 0.999, REF_FS * 1.001, alpha=0.15,
                color=C_REF, zorder=2, label="±0.1% tolerance band")
    for bar, val, model in zip(bars1, fs_vals, models):
        err = abs(val - REF_FS) / REF_FS * 100
        color = C_REF if err < 0.15 else (C_GV0 if err < 5 else C_MISS)
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.012,
                 f"{val:.3f}\n({err:+.1f}%)" if model != "Reference\nsolver"
                 else f"{val:.4f}",
                 ha="center", va="bottom", fontsize=8.5,
                 color="#B71C1C" if err > 5 else "#37474F",
                 fontweight="bold" if err > 5 else "normal")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylim(1.3, 2.15)
    ax1.set_ylabel("Factor of Safety  (FS)", fontsize=10)
    ax1.set_title("FS — All Models vs. Reference", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.yaxis.grid(True, alpha=0.35)
    ax1.set_axisbelow(True)

    # λ bars
    bars2 = ax2.bar(x, lam_vals, width=w, color=colors_lam, zorder=3,
                    edgecolor="white", linewidth=0.8)
    ax2.axhline(REF_LAM, color=C_REF, lw=1.5, ls="--", zorder=4,
                label=f"Reference λ = {REF_LAM:.4f}")
    ax2.axhline(0, color="#546E7A", lw=0.8, ls="-", alpha=0.4, zorder=2)
    ax2.axhspan(REF_LAM * 0.99, REF_LAM * 1.01, alpha=0.15,
                color=C_REF, zorder=2, label="±1% tolerance band")
    for bar, val, model in zip(bars2, lam_vals, models):
        err = abs(val - REF_LAM) / abs(REF_LAM) * 100
        va = "bottom" if val >= 0 else "top"
        offset = 0.008 if val >= 0 else -0.008
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + offset,
                 f"{val:.3f}\n({err:+.0f}%)" if model != "Reference\nsolver"
                 else f"{val:.4f}",
                 ha="center", va=va, fontsize=8.5,
                 color="#B71C1C" if err > 10 else "#37474F",
                 fontweight="bold" if err > 10 else "normal")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.set_ylim(-0.28, 0.38)
    ax2.set_ylabel("Interslice scale factor  (λ)", fontsize=10)
    ax2.set_title("λ — All Models vs. Reference", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.yaxis.grid(True, alpha=0.35)
    ax2.set_axisbelow(True)

    fig.suptitle("LLM Benchmark Results — Morgenstern-Price Reference Case",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT, "fig2_llm_benchmark.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Ψ transfer-coefficient degeneracy
# ─────────────────────────────────────────────────────────────────────────────
def fig_psi_degeneracy():
    boundaries = np.array([1, 2, 3, 4])   # j = 1..4 (between slices 1–2, 2–3, 3–4, 4–5)
    psi_llm    = np.ones(4)                # all LLMs: Ψ = 1 everywhere

    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.plot(boundaries, PSI_REF, "o-", color=C_REF, lw=2.5, ms=8,
            zorder=5, label="Reference solver (Zhu eq. 3.54, mixed-index)")
    ax.fill_between(boundaries, PSI_REF, 1.0, alpha=0.12, color=C_REF, zorder=2)

    ax.plot(boundaries, psi_llm, "s--", color=C_MISS, lw=2.0, ms=8,
            zorder=5, label="All tested LLMs: Ψ ≡ 1 (same-slice degeneracy)")

    ax.axhline(1.0, color="#B0BEC5", lw=0.8, ls="-", zorder=1)

    for j, psi in zip(boundaries, PSI_REF):
        ax.annotate(f"Ψ{j} = {psi:.4f}",
                    xy=(j, psi), xytext=(j, psi + 0.018),
                    ha="center", fontsize=8.5, color=C_REF, fontweight="bold")

    ax.annotate("Ψ ≡ 1\n(all LLMs,\nall boundaries)",
                xy=(2.5, 1.0), xytext=(2.7, 1.05),
                fontsize=9, color=C_MISS, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_MISS, lw=1.2))

    ax.set_xticks(boundaries)
    ax.set_xticklabels(
        ["j=1\n(slices 1→2)", "j=2\n(slices 2→3)",
         "j=3\n(slices 3→4)", "j=4\n(slices 4→5)"],
        fontsize=9)
    ax.set_ylim(0.82, 1.12)
    ax.set_xlabel("Slice boundary  j", fontsize=10)
    ax.set_ylabel("Transfer coefficient  Ψⱼ", fontsize=10)
    ax.set_title("Mode 3: Ψ Same-Slice Degeneracy — Universal LLM Failure",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.yaxis.grid(True, alpha=0.3)

    note = ("With f(x) = const., LLM formula collapses numerator = denominator → Ψ = 1  ∀ j.\n"
            "Reference uses mixed indexing: numerator ∝ α_{j+1}, denominator = Φ_j → Ψ ≠ 1.")
    ax.text(0.5, -0.19, note, transform=ax.transAxes,
            ha="center", fontsize=8.5, color="#546E7A",
            style="italic", wrap=True)

    fig.tight_layout()
    path = os.path.join(OUT, "fig3_psi_degeneracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Convergence history (F and λ)
# ─────────────────────────────────────────────────────────────────────────────
def fig_convergence():
    datasets = [
        ("Reference solver", REF_CONV_F,  REF_CONV_LAM,  C_REF,  "o-",  2.5),
        ("Gemini V0",        GEM_V0_F,    GEM_V0_LAM,    C_GV0,  "s--", 1.8),
        ("Gemini V1",        GEM_V1_F,    GEM_V1_LAM,    C_GV1,  "^--", 1.8),
        ("DeepSeek V0",      DS_V0_F,     DS_V0_LAM,     C_DV0,  "D--", 1.8),
        ("DeepSeek V1",      DS_V1_F,     DS_V1_LAM,     C_DV1,  "v--", 1.8),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=False)

    for label, fs_conv, _, color, style, lw in datasets:
        iters = np.arange(1, len(fs_conv) + 1)
        ax1.plot(iters, fs_conv, style, color=color, lw=lw, ms=6.5,
                 label=label, zorder=4)

    ax1.axhline(REF_FS, color=C_REF, lw=1.0, ls=":", alpha=0.5, zorder=3)
    ax1.set_ylabel("Factor of Safety  F", fontsize=10)
    ax1.set_title("Convergence History — Factor of Safety (F)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8.5, loc="lower right", ncol=2)
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_ylim(0.8, 2.15)
    ax1.annotate(f"Target: FS = {REF_FS:.4f}", xy=(1, REF_FS),
                 xytext=(3.5, REF_FS + 0.07), fontsize=8, color=C_REF,
                 arrowprops=dict(arrowstyle="->", color=C_REF, lw=1))

    for label, _, lam_conv, color, style, lw in datasets:
        iters = np.arange(1, len(lam_conv) + 1)
        ax2.plot(iters, lam_conv, style, color=color, lw=lw, ms=6.5,
                 label=label, zorder=4)

    ax2.axhline(REF_LAM, color=C_REF, lw=1.0, ls=":", alpha=0.5, zorder=3)
    ax2.axhline(0.0, color="#B0BEC5", lw=0.7, ls="-", zorder=1)
    ax2.set_xlabel("Iteration", fontsize=10)
    ax2.set_ylabel("Scale factor  λ", fontsize=10)
    ax2.set_title("Convergence History — Scale Factor (λ)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8.5, loc="lower right", ncol=2)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_ylim(-0.25, 0.35)
    ax2.annotate(f"Target: λ = {REF_LAM:.4f}", xy=(1, REF_LAM),
                 xytext=(3.5, REF_LAM + 0.04), fontsize=8, color=C_REF,
                 arrowprops=dict(arrowstyle="->", color=C_REF, lw=1))

    for ax in (ax1, ax2):
        ax.set_xticks(range(1, 8))
        ax.set_xticklabels([str(i) for i in range(1, 8)], fontsize=9)
        ax.set_axisbelow(True)

    fig.suptitle("Iteration Convergence — Reference vs. LLM Implementations",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT, "fig4_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Interslice force profile (E and X) with thrust line
# ─────────────────────────────────────────────────────────────────────────────
def fig_interslice_forces():
    boundaries = np.arange(6)
    labels = ["E₀\n(left)", "E₁", "E₂", "E₃", "E₄", "E₅\n(right)"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # E distribution
    ax1 = axes[0]
    ax1.bar(boundaries, E_REF, color=C_REF, alpha=0.75, zorder=3,
            edgecolor="white", linewidth=0.8)
    ax1.plot(boundaries, E_REF, "o-", color=C_REF, lw=1.8, ms=6, zorder=4)
    for i, (b, e) in enumerate(zip(boundaries, E_REF)):
        if e > 10:
            ax1.text(b, e + 40, f"{e:.0f}", ha="center", fontsize=8.5,
                     color="#1B5E20", fontweight="bold")
        else:
            ax1.text(b, e + 40, f"{e:.0f}", ha="center", fontsize=8.5,
                     color="#546E7A")
    ax1.set_xticks(boundaries)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("E  (kN/m)", fontsize=10)
    ax1.set_title("Interslice Normal Forces  E", fontsize=11, fontweight="bold")
    ax1.set_ylim(-100, 3600)
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.annotate("Boundary conditions:\nE₀ = E₅ = 0  (no external horizontal load)",
                 xy=(0.5, 0.91), xycoords="axes fraction",
                 ha="center", fontsize=8.5, color="#546E7A",
                 bbox=dict(fc="#F5F5F5", ec="#BDBDBD", boxstyle="round,pad=0.3"))

    # X and thrust-line z = X/E = λ
    ax2 = axes[1]
    ax2.bar(boundaries, X_SHEAR, color="#0277BD", alpha=0.65, zorder=3,
            edgecolor="white", linewidth=0.8, label="X = λ·f·E  (shear)")
    ax2.plot(boundaries, X_SHEAR, "o-", color="#0277BD", lw=1.8, ms=6, zorder=4)
    for i, (b, x) in enumerate(zip(boundaries, X_SHEAR)):
        if x > 10:
            ax2.text(b, x + 10, f"{x:.0f}", ha="center", fontsize=8.5,
                     color="#01579B", fontweight="bold")

    ax2_r = ax2.twinx()
    z_vals = np.full(6, REF_LAM)   # z = X/E = λ (constant for f=const)
    # endpoints undefined (E=0), show only interior
    ax2_r.plot(boundaries[1:-1], z_vals[1:-1], "D--",
               color="#E65100", lw=1.8, ms=7, zorder=5,
               label=f"Thrust line z = X/E = λ = {REF_LAM:.4f}")
    ax2_r.set_ylim(0, 0.6)
    ax2_r.set_ylabel("Thrust-line height  z = X/E  (m/m)", fontsize=9,
                     color="#E65100")
    ax2_r.tick_params(axis="y", colors="#E65100", labelsize=9)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               fontsize=8.5, loc="upper right")

    ax2.set_xticks(boundaries)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("X  (kN/m)", fontsize=10)
    ax2.set_title("Interslice Shear Forces  X  +  Thrust Line", fontsize=11, fontweight="bold")
    ax2.set_ylim(-50, 850)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    fig.suptitle(f"Interslice Force Distribution — FS = {REF_FS:.4f}, λ = {REF_LAM:.4f}",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT, "fig5_interslice_forces.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...")
    fig_slope_geometry()
    fig_llm_benchmark()
    fig_psi_degeneracy()
    fig_convergence()
    fig_interslice_forces()
    print(f"\nAll figures saved to ./{OUT}/")
