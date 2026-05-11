"""
inputs.py — Morgenstern-Price Solver
Input parameters for slope stability analysis.

Reference: Example 1, Case 1 from the dissertation
"Análise de Estabilidade de Taludes pelos Métodos de Morgenstern-Price e Correia"
following the Zhu et al. (2005) algorithm.

Benchmark (5 slices, dry, no seismic, f(x) = constant):
    FS ≈ 1.971,  λ ≈ 0.316  (Slope/Taludes_V13)
    FS ≈ 2.031             (Slide — larger deviation at low n)

NOTE: γ = 16 kN/m³ as stated in PDF Quadro 4.1.
      The original task spec listed 18 kN/m³, which is inconsistent
      with the benchmark. 16 kN/m³ is required to reproduce FS = 1.971.
"""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class SlopeGeometry:
    """
    Geometric definition of the slope surface (linear) and circular slip surface.
    Coordinate system: x = horizontal (left → right), y = vertical (down → up).
    """

    # ── Slope surface: two endpoints of the linear slope segment ──────────────
    x_crest: float = 10.0     # [m] x of crest (top of slope face)
    y_crest: float = 46.0     # [m] y of crest
    x_toe:   float = 134.0    # [m] x of toe   (bottom of slope face)
    y_toe:   float = 15.0     # [m] y of toe

    # ── Circular slip surface ─────────────────────────────────────────────────
    x_center: float = 95.0    # [m] x of circle center
    y_center: float = 120.0   # [m] y of circle center
    radius:   float = 112.0   # [m] radius R


@dataclass
class SoilParams:
    """
    Effective Mohr-Coulomb parameters for a homogeneous, isotropic, dry soil mass.
    No pore pressure (u = 0), no seismic loading (Kh = Kv = 0).
    """
    cohesion:     float = 12.5   # c'  [kPa]
    friction_deg: float = 20.0   # φ'  [degrees]
    unit_weight:  float = 16.0   # γ   [kN/m³]  — confirmed from PDF Quadro 4.1

    @property
    def friction_rad(self) -> float:
        """Effective friction angle in radians."""
        return math.radians(self.friction_deg)

    @property
    def tan_phi(self) -> float:
        return math.tan(self.friction_rad)


@dataclass
class NumericalParams:
    """
    Solver configuration.

    Initial values follow Zhu et al. (2005) recommendation:
        F₀ = 1.0,  λ₀ = 0.0
    The dissertation reports 4 iterations for convergence on this example
    with tolerance = 1e-5.

    f_function:
        "constant"  → f(x) = 1 for all x  (Spencer method, no λ coupling to shape)
        "half_sine" → f(x) = sin(π·(x−a)/(b−a))  (standard half-sine)
    """
    n_slices:    int   = 5
    fs_init:     float = 1.0       # F₀ — Zhu's recommended starting point
    lambda_init: float = 0.0       # λ₀ — Zhu's recommended starting point
    tolerance:   float = 1e-5      # convergence tolerance (dissertation uses 1e-5 for M-P)
    max_iter:    int   = 100
    f_function:  str   = "constant"  # "constant" | "half_sine"
    verbose:     bool  = True


def default_inputs() -> tuple[SlopeGeometry, SoilParams, NumericalParams]:
    """Return the default benchmark inputs as a (geometry, soil, numerical) tuple."""
    return SlopeGeometry(), SoilParams(), NumericalParams()
