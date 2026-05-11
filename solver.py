"""
solver.py — Morgenstern-Price Solver
Iterative solver following the Zhu et al. (2005) algorithm.

Mathematical reference:
  Dissertation eqs. 3.43–3.59:
    R̄_i = W_i·cos(α_i) − u_i                                        [3.44]
    R_i  = R̄_i·tan(φ') + c'·b_i/cos(α_i)                           [3.50]
    T_i  = W_i·sin(α_i)                                               [3.48]
    Φ_i  = (sin αᵢ − λ·fᵣ_i·cos αᵢ)·tan φ' + (cos αᵢ + λ·fᵣ_i·sin αᵢ)·F  [3.52]
    Ψ_j  = [(sin α_{j+1} − λ·f_j·cos α_{j+1})·tan φ'
            + (cos α_{j+1} + λ·f_j·sin α_{j+1})·F] / Φ_j             [3.54]
    F    = Σ(R_i·P_i) / Σ(T_i·P_i),  P_i = ∏_{j=i}^{n-1} Ψ_j       [3.56]
    λ    = Σ[b_i·(E_i+E_{i-1})·tan αᵢ] / Σ[b_i·(fᵣ_i·E_i+fₗ_i·E_{i-1})]  [3.59]

Simplifications applied (dry slope, no seismic, no surcharge):
    u_i = 0,  Kh = Kv = 0,  Q = 0

Sign convention for α (base inclination):
    α > 0 on the right/passive side  (x_mid > x_center)
    α < 0 on the left/active  side   (x_mid < x_center)

Known LLM failure modes for this problem (documented for AI evaluation):
  1. Confusing Bishop's N'_i formula with M-P's full interslice formulation
  2. Using only moment equilibrium (missing the force equilibrium equation)
  3. Wrong sign on α for the active zone → FS diverges
  4. Using scipy.optimize as a black-box instead of the explicit Zhu recurrence
  5. Ignoring Φ_i in the denominator of Ψ (setting it to 1)
  6. Confusing f_left/f_right indexing in the Ψ transfer coefficient
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List

import numpy as np

from inputs import SoilParams, NumericalParams
from discretizer import SliceData


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class MPResult:
    """Complete output of the Morgenstern-Price solver."""
    FS:             float          # Factor of safety
    lam:            float          # Scale factor λ
    n_iter:         int            # Iterations to convergence
    converged:      bool           # True if tolerance was met
    E:              np.ndarray     # [n+1]  interslice normal forces [kN/m], E[0]=E[n]=0
    X:              np.ndarray     # [n+1]  interslice shear forces = λ·f·E [kN/m]
    N_prime:        np.ndarray     # [n]    effective normal force at base [kN/m]
    T_res:          np.ndarray     # [n]    available shear resistance [kN/m]
    T_mob:          np.ndarray     # [n]    mobilized shear at base [kN/m]
    sigma_n:        np.ndarray     # [n]    normal stress at base [kPa]
    residual_force: float          # E[n]  — global horizontal force residual (≈ 0)


# ── Solver class ───────────────────────────────────────────────────────────────

class MPSolver:
    """
    Morgenstern-Price solver using the Zhu et al. (2005) explicit iteration.

    The algorithm does NOT use scipy.optimize. It uses the closed-form
    expressions for F (eq. 3.56) and λ (eq. 3.59) derived by Zhu et al. from
    the force and moment equilibrium conditions.  The two unknowns (F, λ) are
    updated alternately until both converge.

    Iteration count is typically 3–4 for the benchmark case (dissertation
    reports exactly 4 iterations for Case 1, f = constant).
    """

    def __init__(
        self,
        slices: List[SliceData],
        soil: SoilParams,
        num: NumericalParams,
    ) -> None:
        self.slices  = slices
        self.soil    = soil
        self.num     = num
        self.n       = len(slices)

        # ── Extract arrays (0-based, left → right) ────────────────────────────
        self.alpha = np.array([s.alpha for s in slices])   # base inclination [rad]
        self.W     = np.array([s.W     for s in slices])   # weight [kN/m]
        self.b     = np.array([s.b     for s in slices])   # width  [m]
        self.u     = np.array([s.u     for s in slices])   # pore pressure [kPa]

        # f_boundary[k] = f(x) at vertical boundary k, k = 0..n
        #   k = 0:  left edge of slice 1
        #   k = i:  right edge of slice i  =  left edge of slice i+1
        #   k = n:  right edge of slice n
        self.f_boundary = np.array(
            [slices[0].f_left] + [s.f_right for s in slices]
        )   # shape [n+1]

        self.tan_phi = soil.tan_phi
        self.c       = soil.cohesion

    # ── Pre-computed quantities (F, λ independent) ────────────────────────────

    def _precompute_RT(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        R̄_i, R_i, T_i — computed once before the iteration loop.

        R̄_i = W_i·cos(α_i) − u_i               [eq. 3.44 simplified]
        R_i  = R̄_i·tan(φ') + c'·b_i/cos(α_i)   [eq. 3.50]
        T_i  = W_i·sin(α_i)                      [eq. 3.48 simplified]
        """
        cos_a = np.cos(self.alpha)
        sin_a = np.sin(self.alpha)

        Rbar = self.W * cos_a - self.u
        R    = Rbar * self.tan_phi + self.c * self.b / cos_a
        T    = self.W * sin_a
        return Rbar, R, T

    # ── F/λ-dependent quantities ──────────────────────────────────────────────

    def _compute_Phi(self, F: float, lam: float) -> np.ndarray:
        """
        Φ_i for each slice i [eq. 3.52]:
            Φ_i = (sin αᵢ − λ·fᵣᵢ·cos αᵢ)·tan φ' + (cos αᵢ + λ·fᵣᵢ·sin αᵢ)·F

        fᵣᵢ = f_boundary[i+1]  (right face of 0-based slice i)
        Returns array of shape [n].
        """
        sin_a   = np.sin(self.alpha)
        cos_a   = np.cos(self.alpha)
        f_right = self.f_boundary[1:]      # right-face values, shape [n]

        return ((sin_a - lam * f_right * cos_a) * self.tan_phi
                + (cos_a + lam * f_right * sin_a) * F)

    def _compute_Psi(self, F: float, lam: float, Phi: np.ndarray) -> np.ndarray:
        """
        Ψ_j for j = 1..n-1  [eq. 3.54]:
        Transfer coefficient at boundary j (between 1-based slices j and j+1):

            Ψ_j = [(sin α_{j+1} − λ·f_j·cos α_{j+1})·tan φ'
                   + (cos α_{j+1} + λ·f_j·sin α_{j+1})·F] / Φ_j

        Index mapping (0-based):
          j = 1..n-1  →  array index k = j-1 = 0..n-2
          α_{j+1}  = alpha[j]     (0-based)
          f_j      = f_boundary[j] (0-based, boundary between slices j and j+1)
          Φ_j      = Phi[j-1]      (0-based)

        Returns array of shape [n-1].

        NOTE: The numerator uses α_{j+1} (NEXT slice's angle) while the
        denominator uses Φ_j (computed for the CURRENT slice's angle α_j).
        This mixed indexing is standard in Zhu's formulation and is a common
        source of implementation errors.
        """
        sin_next = np.sin(self.alpha[1:])    # α of slices 2..n (0-based: alpha[1..n-1])
        cos_next = np.cos(self.alpha[1:])
        f_j      = self.f_boundary[1:-1]     # f at internal boundaries 1..n-1, shape [n-1]
        Phi_j    = Phi[:-1]                  # Φ of slices 1..n-1 (0-based: Phi[0..n-2])

        numerator = ((sin_next - lam * f_j * cos_next) * self.tan_phi
                     + (cos_next + lam * f_j * sin_next) * F)

        # Guard against Φ_j ≈ 0 (numerical instability)
        safe_Phi = np.where(np.abs(Phi_j) > 1e-12,
                            Phi_j,
                            np.sign(Phi_j + 1e-300) * 1e-12)
        return numerator / safe_Phi

    def _compute_F_new(
        self,
        R: np.ndarray,
        T: np.ndarray,
        Psi: np.ndarray,
    ) -> float:
        """
        Compute the new FS from eq. 3.56:
            F = Σ(R_i · P_i) / Σ(T_i · P_i)

        P_i = ∏_{j=i}^{n-1} Ψ_j   (cumulative right-to-left product)
        P_n  = 1  (empty product)

        0-based: Psi[k] = Ψ_{k+1}  for k = 0..n-2
          P[n-1] = 1
          P[k]   = Psi[k] · P[k+1]   for k = n-2 down to 0
        """
        n = self.n
        P = np.ones(n)
        for k in range(n - 2, -1, -1):
            P[k] = Psi[k] * P[k + 1]

        sum_R = float(np.dot(R, P))
        sum_T = float(np.dot(T, P))

        if abs(sum_T) < 1e-12:
            raise ValueError(
                "Denominator Σ(T_i·P_i) ≈ 0.  "
                "All driving forces vanish — check that the slip surface "
                "is not fully in the passive zone."
            )
        return sum_R / sum_T

    def _compute_E(
        self,
        F: float,
        R: np.ndarray,
        T: np.ndarray,
        Phi: np.ndarray,
        Psi: np.ndarray,
    ) -> np.ndarray:
        """
        Compute interslice normal forces E[0..n] by forward recurrence (eq. 3.55).

        Defining A_i = E_i · Φ_i:
            A_0 = 0  (since E_0 = 0)
            A_i = Ψ_{i-1} · A_{i-1} + F·T_i − R_i
            E_i = A_i / Φ_i

        Boundary check: E_n should ≈ 0 (right-edge condition).
        Non-zero E_n indicates that the current F and λ do not simultaneously
        satisfy both equilibrium conditions — acceptable mid-iteration.

        Returns E of shape [n+1].
        """
        E     = np.zeros(self.n + 1)
        A_prev = 0.0   # A_0 = E_0 · Φ_0 = 0

        for k in range(self.n):
            if k == 0:
                A_k = F * T[k] - R[k]
            else:
                A_k = Psi[k - 1] * A_prev + F * T[k] - R[k]

            phi_k = Phi[k]
            if abs(phi_k) < 1e-12:
                raise ValueError(
                    f"Numerical instability: Φ_{k+1} = {phi_k:.3e} ≈ 0.  "
                    "The slice equilibrium is singular. Try a different "
                    "initial (F₀, λ₀) or check the slope geometry."
                )
            E[k + 1] = A_k / phi_k
            A_prev   = A_k

        return E

    def _compute_lambda_new(self, E: np.ndarray) -> float:
        """
        Compute the new λ from moment equilibrium (eq. 3.59).
        No seismic/surcharge terms:

            λ = Σ_i[b_i·(E_i + E_{i-1})·tan αᵢ]
                ─────────────────────────────────────────────────────────
                Σ_i[b_i·(fᵣᵢ·E_i + fₗᵢ·E_{i-1})]

        For slice i (1-based):
          E_i    = E[i]  (right boundary)
          E_{i-1}= E[i-1] (left boundary)
          fᵣᵢ = f_boundary[i]
          fₗᵢ = f_boundary[i-1]

        In 0-based slice index k:
          E_right = E[k+1],  E_left = E[k]
          f_right = f_boundary[k+1], f_left = f_boundary[k]
        """
        E_left  = E[:-1]                  # shape [n]
        E_right = E[1:]                   # shape [n]
        f_left  = self.f_boundary[:-1]    # shape [n]
        f_right = self.f_boundary[1:]     # shape [n]
        tan_a   = np.tan(self.alpha)

        numerator   = float(np.dot(self.b * (E_right + E_left), tan_a))
        denominator = float(np.dot(self.b, f_right * E_right + f_left * E_left))

        if abs(denominator) < 1e-12:
            # Denominator is zero when all interslice forces vanish (first iteration
            # with λ = 0 is fine — λ_new ≈ 0 is the correct behaviour).
            return 0.0
        return numerator / denominator

    def _compute_N_prime(
        self,
        Rbar: np.ndarray,
        E: np.ndarray,
        lam: float,
    ) -> np.ndarray:
        """
        Effective normal force at the base of each slice [eq. 3.45]:
            N'_i = R̄_i + E_i·(sin αᵢ − λ·fᵣᵢ·cos αᵢ)
                       − E_{i-1}·(sin αᵢ − λ·fₗᵢ·cos αᵢ)
        """
        sin_a   = np.sin(self.alpha)
        cos_a   = np.cos(self.alpha)
        f_left  = self.f_boundary[:-1]
        f_right = self.f_boundary[1:]
        E_left  = E[:-1]
        E_right = E[1:]

        return (Rbar
                + E_right * (sin_a - lam * f_right * cos_a)
                - E_left  * (sin_a - lam * f_left  * cos_a))

    # ── Main solve method ──────────────────────────────────────────────────────

    def solve(self) -> MPResult:
        """
        Run the Zhu et al. (2005) iterative algorithm to convergence.

        Returns an MPResult with the full solution.
        """
        Rbar, R, T = self._precompute_RT()
        F   = float(self.num.fs_init)
        lam = float(self.num.lambda_init)

        if self.num.verbose:
            print()
            print("=" * 70)
            print("  MORGENSTERN-PRICE SOLVER  —  Zhu et al. (2005) algorithm")
            print("=" * 70)
            print(f"  Slices: {self.n}   |   f(x): {self.num.f_function!r}   |   "
                  f"tol: {self.num.tolerance:.0e}")
            print(f"  Initial guess:  F₀ = {F:.4f},  λ₀ = {lam:.4f}")
            print("-" * 70)
            print(f"  {'Iter':>4}  {'F':>12}  {'λ':>12}  {'|ΔF|':>10}  {'|Δλ|':>10}")
            print("-" * 70)

        converged = False
        n_iter    = 0

        for iteration in range(1, self.num.max_iter + 1):
            n_iter = iteration

            Phi = self._compute_Phi(F, lam)

            if self.n > 1:
                Psi = self._compute_Psi(F, lam, Phi)
            else:
                Psi = np.array([])

            F_new   = self._compute_F_new(R, T, Psi)
            E       = self._compute_E(F_new, R, T, Phi, Psi)
            lam_new = self._compute_lambda_new(E)

            delta_F   = abs(F_new   - F)
            delta_lam = abs(lam_new - lam)

            if self.num.verbose:
                print(f"  {iteration:>4}  {F_new:>12.7f}  {lam_new:>12.7f}  "
                      f"{delta_F:>10.2e}  {delta_lam:>10.2e}")

            F   = F_new
            lam = lam_new

            if delta_F < self.num.tolerance and delta_lam < self.num.tolerance:
                converged = True
                break

        if self.num.verbose:
            print("-" * 70)
            status = "CONVERGED" if converged else "NOT CONVERGED (max iter reached)"
            print(f"  {status}  in {n_iter} iteration(s).")
            print("=" * 70)

        # ── Final quantities at converged (F, λ) ──────────────────────────────
        Phi = self._compute_Phi(F, lam)
        Psi = self._compute_Psi(F, lam, Phi) if self.n > 1 else np.array([])
        E   = self._compute_E(F, R, T, Phi, Psi)

        N_prime = self._compute_N_prime(Rbar, E, lam)

        # Available shear resistance (total): R_i = N'_i·tan(φ') + c'·b_i/cos(α_i)
        T_res = N_prime * self.tan_phi + self.c * self.b / np.cos(self.alpha)
        # Mobilized shear at base = T_res / FS
        T_mob = T_res / F

        # Interslice shear: X_i = λ · f(boundary_i) · E_i
        X = lam * self.f_boundary * E

        # Normal stress at the base mid-point: σ_n = N'_i / (b_i / cos α_i)
        base_area = self.b / np.cos(self.alpha)   # base length per unit depth [m]
        sigma_n = N_prime / base_area             # [kPa]

        return MPResult(
            FS=F,
            lam=lam,
            n_iter=n_iter,
            converged=converged,
            E=E,
            X=X,
            N_prime=N_prime,
            T_res=T_res,
            T_mob=T_mob,
            sigma_n=sigma_n,
            residual_force=float(E[-1]),
        )
