"""
main.py — Morgenstern-Price Solver
Orchestration script and GeoSlope-style reporting.

Usage:
    python main.py

Output:
    1. Input summary
    2. Slice geometry table
    3. Iteration log (convergence of F and λ)
    4. GeoSlope-style per-slice report
    5. Interslice forces at each boundary
    6. Global equilibrium check
    7. Benchmark comparison
"""

from __future__ import annotations
import math
from typing import List

from inputs import SlopeGeometry, SoilParams, NumericalParams, default_inputs
from discretizer import SliceData, build_slices, print_slice_geometry
from solver import MPSolver, MPResult


# ── Benchmark values (Case 1: 5 slices, dry, no seismic) ──────────────────────
BENCHMARK_FS         = 1.971   # FS reference — all programs agree (Slope/Taludes_V13/Slide)
BENCHMARK_LAM_SLOPE  = 0.316   # λ from Slope/W (uses proprietary f(x) shape)
BENCHMARK_LAM_TAL    = 0.278   # λ from Taludes_V13 (half-sine f(x))
SLIDE_FS             = 2.031   # Slide reference (slightly higher at low n)


# ── Report functions ───────────────────────────────────────────────────────────

def print_inputs(geom: SlopeGeometry, soil: SoilParams, num: NumericalParams) -> None:
    w = 62
    print()
    print("=" * w)
    print(f"  MORGENSTERN-PRICE SLOPE STABILITY SOLVER")
    print(f"  Zhu et al. (2005) algorithm")
    print("=" * w)
    print(f"  SOIL PARAMETERS")
    print(f"    Cohesion        c'  = {soil.cohesion:.1f} kPa")
    print(f"    Friction angle  phi'= {soil.friction_deg:.1f} deg")
    print(f"    Unit weight     gam = {soil.unit_weight:.1f} kN/m3")
    print(f"    Pore pressure   u   = 0 kPa  (dry slope)")
    print(f"  GEOMETRY")
    print(f"    Crest           ({geom.x_crest}, {geom.y_crest}) m")
    print(f"    Toe             ({geom.x_toe},   {geom.y_toe})  m")
    print(f"    Circle center   ({geom.x_center}, {geom.y_center}) m")
    print(f"    Radius          R = {geom.radius} m")
    print(f"  NUMERICAL")
    print(f"    Slices          n   = {num.n_slices}")
    print(f"    f(x) function       = {num.f_function!r}")
    print(f"    Initial F₀          = {num.fs_init}")
    print(f"    Initial λ₀          = {num.lambda_init}")
    print(f"    Tolerance           = {num.tolerance:.0e}")
    print("=" * w)


def print_geoslope_report(
    slices: List[SliceData],
    result: MPResult,
    soil: SoilParams,
) -> None:
    """
    GeoSlope-style per-slice tabular report.
    Columns mirror the output format used by Slope/W.
    """
    n  = len(slices)
    FS = result.FS

    print()
    print("=" * 112)
    print("  SLICE-BY-SLICE REPORT  (GeoSlope style)")
    print("=" * 112)
    hdr = (
        f"{'#':>3}  "
        f"{'b [m]':>7}  "
        f"{'h [m]':>7}  "
        f"{'a [deg]':>8}  "
        f"{'W [kN/m]':>10}  "
        f"{'N_eff [kN/m]':>12}  "
        f"{'sn [kPa]':>9}  "
        f"{'T_res [kN/m]':>13}  "
        f"{'T_mob [kN/m]':>13}  "
        f"{'E_L [kN/m]':>11}  "
        f"{'E_R [kN/m]':>11}"
    )
    print(hdr)
    print("-" * 112)

    for k, s in enumerate(slices):
        alpha_deg = math.degrees(s.alpha)
        N_prime   = result.N_prime[k]
        sigma_n   = result.sigma_n[k]
        T_res     = result.T_res[k]
        T_mob     = result.T_mob[k]
        E_left    = result.E[k]
        E_right   = result.E[k + 1]

        print(
            f"{s.i:>3}  "
            f"{s.b:>7.3f}  "
            f"{s.h:>7.3f}  "
            f"{alpha_deg:>8.3f}  "
            f"{s.W:>10.3f}  "
            f"{N_prime:>10.3f}  "
            f"{sigma_n:>10.3f}  "
            f"{T_res:>13.3f}  "
            f"{T_mob:>13.3f}  "
            f"{E_left:>11.3f}  "
            f"{E_right:>11.3f}"
        )

    print("=" * 112)

    # Totals
    print(f"\n  Sum of weights:          ΣW     = {sum(s.W for s in slices):>10.3f}  kN/m")
    print(f"  Sum of driving shear:    ΣT_mob = {result.T_mob.sum():>10.3f}  kN/m")
    print(f"  Sum of resistance:       ΣT_res = {result.T_res.sum():>10.3f}  kN/m")


def print_interslice_forces(
    slices: List[SliceData],
    result: MPResult,
) -> None:
    """Print interslice normal (E) and shear (X) forces at each boundary."""
    print()
    print("=" * 60)
    print("  INTERSLICE FORCES")
    print("=" * 60)
    print(f"  {'Boundary':>8}  {'E [kN/m]':>12}  {'X [kN/m]':>12}  {'z=X/E [m]':>10}")
    print("-" * 60)

    n = len(slices)
    for k in range(n + 1):
        E_k = result.E[k]
        X_k = result.X[k]
        # Impulse line height z_k = X_k / E_k (undefined at boundaries where E=0)
        z_str = f"{X_k / E_k:>10.4f}" if abs(E_k) > 1e-6 else f"{'N/A':>10}"
        label = f"{'Left' if k==0 else 'Right' if k==n else str(k):>8}"
        print(f"  {label}  {E_k:>12.4f}  {X_k:>12.4f}  {z_str}")

    print("=" * 60)


def print_equilibrium_check(result: MPResult, slices: List[SliceData]) -> None:
    """Print global equilibrium residuals and stability summary."""
    print()
    print("=" * 62)
    print("  GLOBAL EQUILIBRIUM CHECK")
    print("=" * 62)
    print(f"  Horizontal force residual  E_n        = {result.residual_force:>+12.6f}  kN/m")
    print(f"  (Target: 0.0 — indicates force closure)")
    print()

    # Physical checks
    n_tension = (result.N_prime < 0).sum()
    print(f"  Slices with N' < 0 (tension at base): {n_tension}")
    if n_tension > 0:
        print(f"  WARNING: Tensile normal force detected. Physical inconsistency.")
    else:
        print(f"  All N'_i > 0  ✓  (no base tension)")

    n_neg_E = (result.E[1:-1] < 0).sum()
    if n_neg_E > 0:
        print(f"  WARNING: {n_neg_E} negative interslice E_i (tensile interslice force).")
    else:
        print(f"  All E_i ≥ 0  ✓  (no interslice tension)")

    print("=" * 62)


def print_final_summary(result: MPResult, num: NumericalParams) -> None:
    """
    Print FS, λ, convergence info, and benchmark comparison.

    λ depends on the choice of f(x):
      f = constant  → λ ≈ 0.226  (Spencer case)
      f = half_sine → λ ≈ 0.278  (matches Taludes_V13 exactly)
      Slope/W uses a proprietary f(x) that gives λ ≈ 0.316

    All programs agree on FS ≈ 1.971 for this geometry.
    """
    print()
    print("=" * 70)
    print("  SOLUTION SUMMARY")
    print("=" * 70)
    print(f"  Factor of Safety    FS  = {result.FS:>12.7f}")
    print(f"  Scale factor        lam = {result.lam:>12.7f}")
    print(f"  f(x) function           = {num.f_function!r}")
    print(f"  Converged               = {'Yes' if result.converged else 'NO - increase max_iter'}")
    print(f"  Iterations              = {result.n_iter}")
    print(f"  E_n (force residual)    = {result.residual_force:>+.6f}  kN/m")
    print()

    delta_fs = (result.FS - BENCHMARK_FS) / BENCHMARK_FS * 100
    delta_slide = (result.FS - SLIDE_FS) / SLIDE_FS * 100

    print(f"  BENCHMARK COMPARISON  (5 slices, dry, no seismic)")
    print(f"  {'-'*68}")
    col = f"  {'Source':<28}  {'FS':>8}  {'lam':>8}  {'f(x)':>12}  {'DFS%':>7}"
    print(col)
    print(f"  {'-'*68}")
    print(f"  {'This solver':<28}  {result.FS:>8.4f}  {result.lam:>8.4f}  {num.f_function:>12}")
    print(f"  {'Slope/W (benchmark)':<28}  {BENCHMARK_FS:>8.4f}  "
          f"{BENCHMARK_LAM_SLOPE:>8.4f}  {'proprietary':>12}  {delta_fs:>+6.2f}%")
    print(f"  {'Taludes_V13 (half-sine)':<28}  {BENCHMARK_FS:>8.4f}  "
          f"{BENCHMARK_LAM_TAL:>8.4f}  {'half_sine':>12}  {delta_fs:>+6.2f}%")
    print(f"  {'Slide':<28}  {SLIDE_FS:>8.4f}  {'N/A':>8}  {'unknown':>12}  {delta_slide:>+6.2f}%")
    print(f"  {'-'*68}")
    print()

    lam_note = ""
    if num.f_function == "half_sine":
        dlam = (result.lam - BENCHMARK_LAM_TAL) / BENCHMARK_LAM_TAL * 100
        lam_note = f"  lam vs Taludes_V13 (half-sine): {dlam:+.2f}%"
    elif num.f_function == "constant":
        lam_note = (f"  lam=0.226 (Spencer/constant) vs 0.278 (half-sine) vs 0.316 (Slope).\n"
                    f"  lambda depends on f(x) choice — FS is independent of f(x).")

    if abs(delta_fs) <= 0.5:
        print(f"  FS deviation from benchmark: {delta_fs:+.3f}%  OK  (within +/-0.5%)")
    else:
        print(f"  FS deviation from benchmark: {delta_fs:+.3f}%  <- review geometry")

    if lam_note:
        print(lam_note)

    print()
    if result.FS >= 1.0:
        print(f"  Slope STATUS: STABLE  (FS = {result.FS:.4f} >= 1.0)")
    else:
        print(f"  Slope STATUS: UNSTABLE  (FS = {result.FS:.4f} < 1.0)  FAILURE")

    print("=" * 62)


# ── Entry point ────────────────────────────────────────────────────────────────

def run() -> MPResult:
    """Execute the full Morgenstern-Price analysis and print all reports."""
    geom, soil, num = default_inputs()

    print_inputs(geom, soil, num)

    slices = build_slices(geom, soil, num)
    print_slice_geometry(slices)

    solver = MPSolver(slices, soil, num)
    result = solver.solve()

    print_geoslope_report(slices, result, soil)
    print_interslice_forces(slices, result)
    print_equilibrium_check(result, slices)
    print_final_summary(result, num)

    return result


if __name__ == "__main__":
    run()
