# Morgenstern-Price Slope Stability Solver — LLM Test Prompt (V1)

**Purpose:** Failure-detection benchmark for LLM implementations of Zhu et al. (2005) method.  

---

## PART A — DETERMINISTIC SLOPE STABILITY ANALYSIS

You are a geotechnical engineer implementing the Morgenstern-Price limit-equilibrium method for slope stability using the method defined by Zhu et al. (2005).

### GEOMETRY (x = horizontal, y = vertical)
- Slope crest coordinates: (10.0, 46.0) m
- Slope toe coordinates: (134.0, 15.0) m
- Slip circle center: (95.0, 120.0) m
- Slip circle radius R = 112.0 m

### SOIL PARAMETERS
- Effective cohesion c' = 12.5 kPa
- Effective friction angle φ' = 20.0°
- Unit weight γ = 16.0 kN/m³
- Pore pressure u = 0 kPa (dry slope, no seismic loads)

### NUMERICAL SETTINGS
- Number of slices n = 5 (divided into equal horizontal widths)
- Interslice force function f(x) = 1.0 (constant / Spencer assumption)
- Initial guesses: F₀ = 1.0, λ₀ = 0.0
- Convergence tolerance: 1e-5

---

### TASK A1: Discretization Geometry

- Calculate the entry and exit x-coordinates where the slip circle intersects the slope surface.
- Divide the sliding mass into n=5 equal-width vertical slices.
- For each slice, report: mid-point x_mid, base elevation y_bot, slice height h, slice weight W, and base inclination angle α.
- Report all values in a table format.

---

### TASK A2: Formulate the Equilibrium Equations

Formulate the complete Morgenstern-Price equilibrium equations:

- Formulate the effective normal base force **N'ᵢ** in terms of W, u, α, E (interslice normal forces), and λ, f(x).
- Formulate the **Zhu et al. (2005) explicit equations** used to calculate:
  - **Factor of Safety (F)** from force equilibrium
  - **Scale factor (λ)** from moment equilibrium
  - **Transfer coefficient (Ψⱼ)** and the recurrence relation
- For each equation, provide the exact mathematical formulation with all terms and their definitions.

---

### TASK A3: Execute the Solver

- Execute the iterative procedure until the tolerance (1e-5) is met.
- **Report the iteration log** with all iterations. For each iteration, report:
  - Iteration number
  - F value
  - λ value
  - |ΔF| (magnitude of F change)
  - |Δλ| (magnitude of λ change)
- Report the final interslice normal forces (E) at each slice boundary.
- Report convergence status: converged or diverged.

---

## PART B — METHODOLOGICAL VERIFICATION

Answer the following questions strictly with the exact mathematical formulation or methodological choice you utilized in Part A. Do not explain the reasoning—state the math or the computational method exactly as implemented.

---

### TASK B1: Angle Convention

Write the exact mathematical formula you used to calculate the slice base angle α as a function of the slice midpoint (x_mid), circle center (x_c, y_c), and base elevation (y_bot).

State exactly what the sign of α evaluates to in the active zone (x_mid < x_c) versus the passive zone (x_mid > x_c) based on your formula.

---

### TASK B2: Solver Strategy

Did you treat F and λ as a system of equations solved via a general-purpose non-linear root finder (e.g., scipy.optimize.fsolve), or did you use the exact explicit recurrence relations derived by Zhu et al. (2005)?

State the exact numerical strategy used.

---

### TASK B3: Base Normal Force

Write the exact formula you used for the effective normal force **N'ᵢ** at the base of the slice.

Explicitly state whether your formulation of N'ᵢ includes the interslice shear forces (X) and interslice normal forces (E), and how.

---

### TASK B4: Transfer Coefficient (Ψ)

Write your exact formula for the transfer coefficient Ψⱼ used to link slice j and slice j+1. Show both the numerator and the denominator clearly.

Then evaluate Ψ at j=2 using the numerical values from your solution:
- Numerator at j=2: ___
- Denominator at j=2: ___
- Ψ₂ numerical value: ___

---

### TASK B5: Interslice Function Indexing

In the equation used to calculate the new λ from moment equilibrium, you must evaluate the interslice function f(x). For a given slice 'k', state exactly which boundary locations of f(x) are evaluated and how they multiply the normal forces E.

---

## PART C — FINAL SUMMARY

Based on your complete execution of the Morgenstern-Price algorithm:

1. **What is the final converged Factor of Safety (FS)?**
2. **What is the final converged scale factor (λ)?**
3. **What is the horizontal force residual (E_n) at the final boundary?**

### Observations

- Report the sign of α for each slice and explain whether each slice contributes positively or negatively to the driving shear term.
- State explicitly whether E_j is computed forward or backward through the slices.
- Report the maximum absolute equilibrium residual at the final iteration (or the residual norm used to assess convergence).

---

## Notes for Responder

- Provide **direct answers** with minimal conversational filler.
- Use **exact mathematical notation** matching the Zhu et al. (2005) formulation.
- **Do not approximate** or simplify equations; state them in complete form.
- If you use a published equation, cite the equation number from the dissertation or referenced work.
