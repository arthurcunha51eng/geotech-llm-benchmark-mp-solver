Act strictly based on the parameters and instructions provided in this prompt. Do not use external prior assumptions about typical soil parameters or typical default code libraries. Provide direct answers, equations, and numerical values without excessive conversational filler.

PART A — DETERMINISTIC SLOPE STABILITY ANALYSIS
You are a geotechnical engineer implementing the Morgenstern-Price limit-equilibrium method for slope stability. You must use the method defined by Zhu et al. (2005).

GEOMETRY (x = horizontal, y = vertical):
Slope crest coordinates: (10.0, 46.0) m
Slope toe coordinates: (134.0, 15.0) m
Slip circle center: (95.0, 120.0) m
Slip circle radius R = 112.0 m

SOIL PARAMETERS:
Effective cohesion c' = 12.5 kPa
Effective friction angle φ' = 20.0°
Unit weight γ = 16.0 kN/m³
Pore pressure u = 0 kPa (dry slope, no seismic loads)

NUMERICAL SETTINGS:
Number of slices n = 5 (divided into equal horizontal widths)
Interslice force function f(x) = 1.0 (constant / Spencer assumption)
Initial guesses: F₀ = 1.0, λ₀ = 0.0

TASK A1: Discretization Geometry
— Calculate the entry and exit x-coordinates where the slip circle intersects the slope surface.
— Divide the sliding mass into n=5 equal-width vertical slices.
— For the active (left) side and passive (right) side slices, report: mid-point x_mid, base elevation y_bot, slice height h, slice weight W, and base inclination angle α.

TASK A2: Formulate the Equilibrium Equations
— Formulate the effective normal base force N'_i.
— Formulate the Zhu et al. (2005) explicit equations used to calculate the Factor of Safety (F) from force equilibrium and the scale factor (λ) from moment equilibrium.
— Formulate the equation for the transfer coefficient Ψ_j.

TASK A3: Execute the Solver
— Execute the iterative procedure until the tolerance (1e-5) is met.
— Report the iteration log (F and λ values per iteration).
— Report the final interslice normal forces (E) at each boundary.


PART B — METHODOLOGICAL VERIFICATION
Answer the following questions strictly with the exact mathematical formulation or methodological choice you utilized in Part A. Do not explain the reasoning, just state the math or the exact computational method.

TASK B1: Angle Convention
Write the exact mathematical formula you used to calculate the slice base angle α as a function of the slice midpoint (x_mid), circle center (x_c, y_c), and base elevation (y_bot). State exactly what the sign of α evaluates to in the active zone (x_mid < x_c) versus the passive zone (x_mid > x_c) based on your formula.

TASK B2: Solver Strategy
Did you treat F and λ as a system of equations solved via a general-purpose non-linear root finder (e.g., scipy.optimize.fsolve), or did you use the exact explicit recurrence relations derived by Zhu et al. (2005)? State the exact method used.

TASK B3: Base Normal Force
Write the exact formula you used for the effective normal force N'_i at the base of the slice. Explicitly state whether your formulation of N'_i includes the interslice shear forces (X) and interslice normal forces (E).

TASK B4: Transfer Coefficient (Ψ)
Write your exact formula for the transfer coefficient Ψ_j used to link slice j and slice j+1. Show both the numerator and the denominator clearly.

TASK B5: Interslice Function Indexing
In the equation used to calculate the new λ from moment equilibrium, you must evaluate the interslice function f(x). For a given slice 'k', state exactly which boundary locations of f(x) are evaluated and how they multiply the normal forces E.


PART C — FINAL SUMMARY
Based on your complete execution of the Morgenstern-Price algorithm:

1. What is the final converged Factor of Safety (FS)?
2. What is the final converged scale factor (λ)?
3. What is the horizontal force residual (E_n) at the final boundary?