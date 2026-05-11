"""
discretizer.py — Morgenstern-Price Solver
Geometry processing: builds the list of vertical slices from slope inputs.

Mathematical basis:
  - Slope surface: piecewise-linear (flat → inclined → flat)
  - Slip surface:  lower arc of a circle with center (x_c, y_c), radius R
  - Intersection:  quadratic equation from substituting y_slope(x) into circle eq.
  - Base angle α:  from the tangent to the circle arc at the slice midpoint
                   tan(α) = (x_mid − x_c) / (y_c − y_bot)
                   Sign convention: α > 0 on right (passive) side,
                                    α < 0 on left  (active)  side.
  - Interslice function f(x): evaluated at each vertical boundary.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import List

from inputs import SlopeGeometry, SoilParams, NumericalParams


# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass
class SliceData:
    """All geometric and mechanical properties of a single vertical slice."""
    i:       int     # 1-based slice index (left → right)
    x_left:  float   # [m] left vertical boundary x
    x_right: float   # [m] right vertical boundary x
    x_mid:   float   # [m] midpoint x
    b:       float   # [m] width  (= x_right − x_left)
    y_top:   float   # [m] slope surface elevation at x_mid
    y_bot:   float   # [m] slip surface elevation at x_mid (lower arc)
    h:       float   # [m] slice height = y_top − y_bot
    alpha:   float   # [rad] base inclination (from circle tangent at x_mid)
    W:       float   # [kN/m] slice weight = γ · h · b
    u:       float   # [kPa] pore pressure at base (= 0, dry slope)
    f_left:  float   # f(x) interslice function value at left boundary
    f_right: float   # f(x) interslice function value at right boundary


# ── Helper functions ───────────────────────────────────────────────────────────

def _slope_y(x: float, geom: SlopeGeometry) -> float:
    """
    Elevation of the slope surface at horizontal position x.
    Piecewise linear:
      x < x_crest  → flat at y_crest
      x_crest ≤ x ≤ x_toe → linear slope
      x > x_toe   → flat at y_toe
    """
    if x <= geom.x_crest:
        return geom.y_crest
    if x >= geom.x_toe:
        return geom.y_toe
    m = (geom.y_toe - geom.y_crest) / (geom.x_toe - geom.x_crest)
    return geom.y_crest + m * (x - geom.x_crest)


def _circle_lower_y(x: float, geom: SlopeGeometry) -> float:
    """
    Elevation of the lower arc of the slip circle at horizontal position x.
    y = y_c − √(R² − (x − x_c)²)
    """
    arg = geom.radius**2 - (x - geom.x_center)**2
    if arg < 0.0:
        raise ValueError(f"x={x:.4f} is outside the slip circle's horizontal extent.")
    return geom.y_center - math.sqrt(arg)


def _base_alpha(x_mid: float, geom: SlopeGeometry) -> float:
    """
    Base inclination angle α at the midpoint of a slice (radians).

    Convention (Zhu et al. 2005 / standard LEM for left-to-right sliding):
        α > 0  on the LEFT  side (x_mid < x_c) — active zone, base dips in
                sliding direction, weight component W·sin(α) drives failure.
        α < 0  on the RIGHT side (x_mid > x_c) — passive zone, weight resists.

    Formula: α = atan2(x_c − x_mid, y_c − y_bot)

    This is the angle of the inward radius (from base point to center) measured
    from the upward vertical.  For the active zone, x_c > x_mid, so atan2 > 0.

    COMMON ERROR: using atan2(x_mid − x_c, ...) gives the opposite sign, which
    makes T = W·sin(α) < 0 on the active side and the FS converges to a negative
    value with the correct magnitude (observed in many LLM implementations).
    """
    y_bot = _circle_lower_y(x_mid, geom)
    return math.atan2(geom.x_center - x_mid, geom.y_center - y_bot)


def _f_interslice(x: float, x_a: float, x_b: float, method: str) -> float:
    """
    Interslice function f(x).

    "constant"  → f(x) = 1.0  (Spencer case)
    "half_sine" → f(x) = sin(π·(x − a)/(b − a))
                  μ=1, υ=1 in Zhu's generalized form (eq. 3.57).
                  Boundary values f(x_a) = f(x_b) = 0 automatically.
    """
    if method == "constant":
        return 1.0
    elif method == "half_sine":
        t = (x - x_a) / (x_b - x_a)
        t = max(0.0, min(1.0, t))   # clamp to [0,1] at boundaries
        return math.sin(math.pi * t)
    else:
        raise ValueError(f"Unknown f_function '{method}'. Use 'constant' or 'half_sine'.")


def _find_circle_slope_intersections(geom: SlopeGeometry) -> tuple[float, float]:
    """
    Find the two x-coordinates where the slip circle intersects the slope line.

    Slope line (between crest and toe):
        y = m·x + q,   m = (y_toe − y_crest)/(x_toe − x_crest)

    Substituting into the circle equation:
        (x − x_c)² + (y − y_c)² = R²
        (x − x_c)² + (m·x + q − y_c)² = R²

    Expanding → A·x² + B·x + C = 0.

    Returns (x_entry, x_exit) with x_entry < x_exit.
    """
    m = (geom.y_toe - geom.y_crest) / (geom.x_toe - geom.x_crest)
    q = geom.y_crest - m * geom.x_crest   # y-intercept of slope line

    xc, yc, R = geom.x_center, geom.y_center, geom.radius
    k = q - yc                             # (q − y_c)

    A = 1.0 + m**2
    B = 2.0 * (m * k - xc)
    C = xc**2 + k**2 - R**2

    discriminant = B**2 - 4.0 * A * C
    if discriminant < 0.0:
        raise ValueError(
            "The slip circle does not intersect the slope line. "
            "Check center/radius coordinates."
        )

    sqrt_disc = math.sqrt(discriminant)
    x1 = (-B - sqrt_disc) / (2.0 * A)
    x2 = (-B + sqrt_disc) / (2.0 * A)
    return (min(x1, x2), max(x1, x2))


# ── Main builder ───────────────────────────────────────────────────────────────

def build_slices(
    geom: SlopeGeometry,
    soil: SoilParams,
    num: NumericalParams,
) -> List[SliceData]:
    """
    Discretize the sliding mass into n equal-width vertical slices.

    Steps:
      1. Find x_entry, x_exit (slip circle ∩ slope surface)
      2. Divide [x_entry, x_exit] into n equal intervals of width b
      3. For each slice, evaluate geometry at x_mid
      4. Compute W_i, α_i, f_left_i, f_right_i

    Returns a list of SliceData (ordered left → right, 1-based index).
    """
    x_entry, x_exit = _find_circle_slope_intersections(geom)
    n = num.n_slices
    b = (x_exit - x_entry) / n

    slices: List[SliceData] = []
    for i in range(1, n + 1):
        x_left  = x_entry + (i - 1) * b
        x_right = x_entry + i * b
        x_mid   = 0.5 * (x_left + x_right)

        y_top = _slope_y(x_mid, geom)
        y_bot = _circle_lower_y(x_mid, geom)
        h     = y_top - y_bot

        if h < 0.0:
            raise ValueError(
                f"Slice {i}: y_top ({y_top:.3f}) < y_bot ({y_bot:.3f}). "
                "The slip surface protrudes above the slope — check geometry."
            )

        alpha = _base_alpha(x_mid, geom)
        W     = soil.unit_weight * h * b
        u     = 0.0   # dry slope, no pore pressure

        f_left  = _f_interslice(x_left,  x_entry, x_exit, num.f_function)
        f_right = _f_interslice(x_right, x_entry, x_exit, num.f_function)

        slices.append(SliceData(
            i=i, x_left=x_left, x_right=x_right, x_mid=x_mid,
            b=b, y_top=y_top, y_bot=y_bot, h=h,
            alpha=alpha, W=W, u=u,
            f_left=f_left, f_right=f_right,
        ))

    return slices


def print_slice_geometry(slices: List[SliceData]) -> None:
    """Print a geometry summary table for the discretized slices."""
    print()
    print("=" * 90)
    print("  SLICE GEOMETRY SUMMARY")
    print("=" * 90)
    header = (
        f"{'#':>3}  {'x_mid':>8}  {'b':>7}  {'y_top':>7}  {'y_bot':>7}  "
        f"{'h':>7}  {'α(°)':>8}  {'W(kN/m)':>10}  {'f_L':>6}  {'f_R':>6}"
    )
    print(header)
    print("-" * 90)
    for s in slices:
        print(
            f"{s.i:>3}  {s.x_mid:>8.3f}  {s.b:>7.3f}  {s.y_top:>7.3f}  {s.y_bot:>7.3f}  "
            f"{s.h:>7.3f}  {math.degrees(s.alpha):>8.3f}  {s.W:>10.3f}  "
            f"{s.f_left:>6.4f}  {s.f_right:>6.4f}"
        )
    print("=" * 90)
    print()


if __name__ == "__main__":
    from inputs import default_inputs

    geom, soil, num = default_inputs()
    slices = build_slices(geom, soil, num)
    print_slice_geometry(slices)
