# LLM Failure Mode Analysis — Morgenstern-Price Benchmark

This document records the empirically observed failure modes produced by large language models
when asked to implement and execute the Morgenstern-Price slope stability method (Zhu et al., 2005)
from scratch.

All outputs are treated as **empirical artifacts**. Failed results are preserved exactly as
produced. No LLM output has been corrected or adjusted here.

**Reference solution (Python implementation, Zhu et al. 2005):** FS = 1.9710, λ = 0.2260 (constant f), 4 iterations.

---

## 1. Models Tested

| Model | Prompt Version | FS | λ | ΔFS% | Δλ% | E_n (kN) | Iterations |
|---|---|---|---|---|---|---|---|
| Gemini 3.1 Pro Preview | V0 | 1.965 | 0.212 | 0.3% | 6.2% | 0.00 | 5 |
| Gemini 3.1 Pro Preview | V1 | 1.881 | 0.053 | 4.6% | 76.5% | 0.00 | 6 |
| DeepSeek v3.0 | V0 | 1.886 | −0.142 | 4.3% | 163% | 0.00 | 7 |
| DeepSeek v3.0 | V1 | 1.542 | 0.234 | 21.7% | 3.5% | −48.2 | 7 |
| **Reference solver** | — | **1.971** | **0.226** | 0% | 0% | **< 1e-5** | **4** |

Threshold for triggering forensic analysis: FS error > 0.1% or λ error > 1%.
All model outputs exceed at least one threshold.

---

## 2. Failure Mode Taxonomy

Six failure modes have been identified across all tests. Modes 1–6 were anticipated from
implementation analysis; Mode 7 was observed experimentally in these runs.

### Mode 1 — Inverted α Sign Convention

**Description:** The base inclination angle is computed with the wrong sign convention,
producing α < 0 in the active (driving) zone and α > 0 in the passive (resisting) zone
— the opposite of the reference.

**Formula used (wrong):**
```
α = arctan((x_mid − x_c) / (y_c − y_bot))   → negative in active zone
α = arcsin((x_mid − x_c) / R)                → negative in active zone
```

**Formula used (correct, reference):**
```
α = atan2(x_c − x_mid, y_c − y_bot)          → positive in active zone
α = arcsin((x_c − x_mid) / R)                → positive in active zone
```

**Effect:** The driving shear term T_i = W_i · sin α_i becomes negative on the active
side. The force equilibrium ∑T_i·P_i flips sign relative to ∑R_i·P_i, and FS converges to
an incorrect (and physically ambiguous) value. λ can become negative.

**Observed in:** DeepSeek V0 and V1 (explicitly stated in B1).

**Diagnostic signal:** λ negative, or FS far from 1.971 with α signs opposite to expected.

---

### Mode 2 — scipy.optimize Black-Box Substitution

**Description:** The model treats (F, λ) as unknowns in a two-equation system and solves
it with a generic Newton/secant solver rather than the explicit Zhu et al. recurrence.

**Statement from DeepSeek V1 (B2):**
> "The full system of two nonlinear equations (E_{n+1}(F,λ)=0 and moment equilibrium = 0)
> was solved using a two-variable secant/Newton solver (equivalent to scipy.optimize.fsolve)."

**Problems:**
- The Zhu et al. algorithm provides explicit closed-form update formulas that converge in
  3–4 iterations. A general-purpose solver adds numerical sensitivity.
- A generic solver may converge to the wrong root, especially when the initial guess is far
  from the solution (F₀ = 1.0).
- The explicit recurrence (eqs. 3.54–3.59) IS the algorithm. Using a black-box substitution
  shows that the model did not internalize the mathematical structure.

**Masking behavior in V0:** In DeepSeek V0, the same solver was described as "secant method
on the force residual" without explicitly mentioning fsolve. The V1 prompt (with the explicit
B2 question) elicited the admission.

**Observed in:** DeepSeek V1 (explicitly declared). Suspected in DeepSeek V0 (described
differently, same numerical results).

**Diagnostic signal:** B2 statement mentioning Newton/secant/residual-based outer loop. E_n ≠ 0.

---

### Mode 3 — Ψ Transfer Coefficient — Same-Slice Degeneracy

**Description:** The transfer coefficient Ψ_j is implemented with both numerator and
denominator referencing the **same slice index** and the same f(x) value. For a constant
interslice function f(x) = 1, this causes Ψ to collapse to exactly 1 everywhere,
eliminating inter-slice coupling from the algorithm.

**Formula (degenerate — observed in both models):**
```
Ψ_j = [D_j + λ·f_j·A_j] / [D_j + λ·f_j·A_j] = 1    (same terms)
```

or equivalently:
```
Ψ_j = [1 + λ·f_{j-1}·A_j] / [1 + λ·f_j·A_j]
     where A_j uses α_j in both numerator and denominator
```

For f(x) = constant: f_{j-1} = f_j, so Ψ = 1.

**Formula (correct, reference):**
The Zhu et al. formula uses **mixed indexing**: the numerator carries terms from slice j+1
(or equivalently α_{j+1}), while the denominator is Φ_j — a different quantity from a
different slice. This ensures Ψ ≠ 1 even for constant f(x).

**Effect:** With Ψ = 1 everywhere, the product P_i = ∏Ψ_j = 1 for all slices.
The FS formula degenerates toward a simpler (Bishop-like) summation, losing the interslice
force propagation. F converges to a value significantly different from 1.971.

**Gemini V0 explicit report (B4):**
```
Ψ₂ numerator: 1.000
Ψ₂ denominator: 1.000
Ψ₂ = 1.000
```

**DeepSeek V0 explicit report (B4):**
> "With f(x)=1 (Spencer assumption), this reduces to Ψ_{i−1} = 1."

**DeepSeek V1 explicit report (B4):**
> "Ψ_i = (D_i + λA_i)/(D_i + λA_i) = 1"

**Observed in:** Gemini V0, Gemini V1, DeepSeek V0, DeepSeek V1. Universal failure.

**Diagnostic signal:** B4 reports Ψ_j = 1 explicitly, or numerator = denominator numerically.

---

### Mode 4 — Boundary Condition Not Enforced (E_n ≠ 0)

**Description:** The final interslice normal force E_n at the right boundary is not zero,
violating the equilibrium condition that no external horizontal forces act on the sliding
mass.

**Observed in:** DeepSeek V1, where E_n = −48.2 kN was reported and attributed to
"residual of force equilibrium (~< 0.1% of total weight)."

**Reference value:** E_n < 1×10⁻⁵ kN (enforced by the explicit recurrence structure).

**Physical interpretation:** A non-zero E_n means an unbalanced horizontal force remains
at the passive boundary. The model acknowledged this but treated it as acceptable precision,
which it is not — the correct algorithm satisfies E_n = 0 identically when the explicit
recurrence is used correctly.

**Root cause:** When a secant/Newton solver is used (Mode 2), the convergence criterion
(max(|ΔF|, |Δλ|) < 1e-5) does not directly enforce E_n = 0. The outer solver can
converge on F and λ without fully satisfying the force balance.

**Observed in:** DeepSeek V1.

**Diagnostic signal:** F7 in forensic questionnaire; C3 observation in primary prompt.

---

### Mode 5 — f_left/f_right Indexing Errors (Theoretical, Not Confirmed Experimentally)

**Description:** The λ update equation requires f evaluated at the **left boundary** (f_left)
and **right boundary** (f_right) of each slice separately. Errors occur when:
- f_right is used where f_left is required (or vice versa)
- A single `f_boundary[i]` is applied to both faces of slice i
- 0-based and 1-based index conventions are mixed

**Status:** This failure was anticipated from static code analysis but was NOT directly
observable in the present experiments, because all tested models used f(x) = 1 (constant),
which makes f_left = f_right = 1 everywhere and masks any indexing errors.

**Detection condition:** This mode becomes detectable only with a non-constant f(x), such as
`half_sine`. The reference solver produces λ = 0.278 for half_sine; an indexing error would
produce a different λ while keeping FS approximately unchanged.

**Diagnostic signal:** B5 question; most visible when testing with half_sine interslice function.

---

### Mode 6 — Unit Weight γ = 18 kN/m³ (Theoretical, Not Triggered)

**Description:** The model substitutes γ = 18 kN/m³ (a common default for dense sand)
instead of the specified γ = 16 kN/m³.

**Status:** Not triggered in these experiments. All tested models correctly used γ = 16
kN/m³ as specified in the prompt.

**Detection condition:** FS ≈ 1.989 instead of 1.971 (approximately +1.0% error from weight
scaling).

---

### Mode 7 — Internally Consistent but Wrong Algorithm (Newly Observed)

**Description:** The model converges to a self-consistent solution (E_n ≈ 0, iteration
residuals → 0) but using an algorithm that is structurally different from Zhu et al. (2005).
The result is plausible-looking but numerically incorrect.

**Gemini V0 specific case:** FS = 1.965 (0.3% from reference), λ = 0.212 (6.2% from
reference). Boundaries satisfied, iteration converged, but the Ψ formula uses same-slice
indexing. The near-correct FS is a numerical coincidence — the F formula without proper Ψ
coupling happens to produce a value close to the reference under this specific geometry.

**Significance:** This is the most dangerous failure mode because:
- The result looks plausible (FS close to reference)
- Standard checks pass (E_n ≈ 0, convergence achieved)
- The error only becomes visible through structural analysis of the Ψ formula (B4)
- Gemini V0 would have passed a naive "FS ≈ 1.97" check

**Diagnostic signal:** B4 evaluation of Ψ at j=2; explicit numerator/denominator values.

---

## 3. Comparative Analysis: V0 vs V1 Prompt

### Gemini

| Metric | V0 | V1 | Change |
|---|---|---|---|
| FS | 1.965 | 1.881 | Worse (+4.3% deviation) |
| λ | 0.212 | 0.053 | Worse (76.5% deviation) |
| α sign | Correct | Correct | No change |
| Ψ value | Near-1 | Exactly 1 (reported) | More visible failure |
| Ψ degeneracy visible | Partial | **Explicit in B4** | **Improvement** |

The V1 prompt produced numerically worse results but **better diagnostic visibility**.
B4 of V1 explicitly reports "Ψ₂ = 1.000", exposing the degeneracy that was only implicit
in V0. This is the intended effect of the refinement: failures emerge more clearly.

The numeric worsening from V0 to V1 may indicate that the V0 formula had partial compensation
(same-index Ψ with slightly different auxiliary term A_j), which the V1 reformulation removed
by explicitly collapsing Ψ to 1.

### DeepSeek

| Metric | V0 | V1 | Change |
|---|---|---|---|
| FS | 1.886 | 1.542 | Worse (+17.4% deviation) |
| λ | −0.142 | +0.234 | Sign flip (both wrong) |
| α sign | Inverted | Inverted | No change |
| Solver strategy | "Secant on residual" | **Admitted fsolve** | **More visible failure** |
| E_n | 0.00 | **−48.2 kN** | Boundary condition exposed |
| Ψ value | Implicit = 1 | Explicitly = 1 | More visible failure |

DeepSeek V0 described its solver as "secant method on the force residual" without using the
word fsolve. V1's explicit B2 question elicited the admission that a secant/Newton solver
(fsolve equivalent) was used. This is a key benefit of the V1 prompt: it exposed a
previously hidden solver strategy.

The E_n = −48.2 kN in V1 is a direct consequence of the fsolve strategy: the outer solver
converges on (F, λ) without enforcing E_n = 0 as a hard constraint, unlike the explicit
recurrence which guarantees it structurally.

---

## 4. Fault Tree

```
FS ≠ 1.971 (deviation > 0.1%)
│
├── α sign inverted? (B1)
│   ├── YES → Mode 1: α convention error
│   │          Formula: arctan((x_mid − x_c)/...) instead of (x_c − x_mid)/...
│   │          Signature: λ < 0, driving shear negative
│   │          Observed: DeepSeek V0, DeepSeek V1
│   └── NO → continue
│
├── Ψ_j = 1 everywhere? (B4)
│   ├── YES → Mode 3: Ψ same-slice degeneracy
│   │          Formula: num and denom reference same slice → Ψ = 1 for f = const
│   │          Signature: P_i = 1 ∀ i, FS not coupled across slices
│   │          Observed: Gemini V0, Gemini V1, DeepSeek V0, DeepSeek V1
│   └── NO → continue
│
├── E_n ≠ 0? (F3/F7)
│   ├── YES → Mode 4: Boundary condition not enforced
│   │          Root cause: outer solver convergence criterion ≠ force balance
│   │          Signature: fsolve used (check B2)
│   │          Observed: DeepSeek V1 (E_n = −48.2 kN)
│   └── NO → continue
│
├── Solver uses external root finder? (B2)
│   ├── YES → Mode 2: Black-box substitution
│   │          Signature: "secant", "Newton", "fsolve equivalent" in B2
│   │          Observed: DeepSeek V1
│   └── NO → continue
│
├── FS close to reference but λ wrong? (> 1% deviation)
│   ├── YES → Mode 5 (suspected): f_left/f_right indexing
│   │          Only detectable with non-constant f(x)
│   │          Cannot confirm from constant-f(x) tests alone
│   └── NO → continue
│
└── Result internally consistent but FS slightly off?
    └── Mode 7: Structurally wrong algorithm producing plausible result
               Requires structural inspection of Ψ formula (B4 explicit values)
               Observed: Gemini V0 (FS = 1.965, near-correct by coincidence)
```

---

## 5. Observations on the Forensic Re-Runs

**Experimental note:** In these tests, the "forensic" run was a re-application of the
**primary benchmark prompt** in the same conversation window (not the F1–F7 structured
questionnaire). The file `M-P_llm_tests/FORENSIC_QUESTIONAIRE.md` contains the same prompt
as PROMPT_V0.md, applied as a continuation of the original conversation.

Both models reproduced their original solutions exactly when the primary prompt was re-run
in the same window. This indicates:

1. **No self-correction from context exposure.** Seeing the primary prompt a second time
   (in the same session) did not cause either model to revise its algorithm.
2. **Internal consistency is stable.** Models that converge to a wrong algorithm maintain
   that algorithm consistently across re-runs within the same session.
3. **The structured F1–F7 forensic questionnaire has not been tested empirically yet.**
   The fault tree analysis in Section 4 and the questionnaire design in the testing
   methodology are based on anticipated diagnostic behavior, not observed empirical runs.

**Design implication:** For future forensic runs, the F1–F7 questionnaire should be applied
in a **fresh session** (no prior context) to avoid conditioning the model on its previous
wrong answers. The V1 prompt improvement (which elicited Ψ = 1 explicitly and fsolve
admission from DeepSeek) demonstrates that structural questions embedded in the primary
prompt can expose failures without a separate forensic pass.

---

## 6. Benchmark Sensitivity Notes

**Why does Ψ degeneracy produce near-correct FS?**

For this specific geometry (5-slice benchmark, c' = 12.5 kPa, φ' = 20°), the slice weights
and angles are arranged such that the F formula without proper Ψ coupling produces F ≈ 1.96
— close to, but not equal to, 1.971. This is a geometric coincidence of the benchmark, not a
feature of the method.

A different geometry (steeper slope, more slices, non-circular slip surface) would be expected
to produce larger deviations under the same Ψ degeneracy. The benchmark case is therefore
**less sensitive than ideal** to Mode 3 failures.

**Why is λ much more sensitive than FS?**

λ is derived from moment equilibrium, which directly couples interslice forces E and the
interslice function f(x). Even small errors in Ψ propagate into the E distribution, which
then propagates into λ through the moment equation. FS (from force equilibrium) is more
robust to E distribution errors because it involves a ratio of sums — individual E errors
partially cancel.

This explains why Gemini achieved FS = 1.965 (−0.3%) while λ = 0.212 (−6.2%):
the force equilibrium is more forgiving of Ψ errors than the moment equilibrium.

---

## 7. Limitations of This Analysis

- Only two models were tested.
- Both models were tested on only one benchmark case (5 slices, dry, no seismic).
- The forensic questionnaire was applied in the same conversation window as the primary
  prompt, not in a fresh session. Prior context may have influenced the responses.
- f(x) = constant was used in all primary runs. Mode 5 (f_left/f_right indexing) cannot
  be confirmed or denied from these data.
- Benchmark case may be geometrically coincidental in its sensitivity (see Section 6).
- Results are empirical observations, not proofs. Each model's output should be treated
  as a data point, not a representative sample of model capability.

---

## 8. Testing Files

| File | Content |
|---|---|
| `M-P_llm_tests/prompts/PROMPT_V0.md` | Original benchmark prompt |
| `M-P_llm_tests/prompts/PROMPT_V1_CLEAN.md` | Refined prompt (explicit B4 eval + iteration log) |
| `M-P_llm_tests/FORENSIC_QUESTIONAIRE.md` | Primary prompt re-run used as forensic in V0/V1 sessions |
| `M-P_llm_tests/results/GEMINI_3.1_PRO_PREVIEW_RESULTS_PROMPT_V0.txt` | Gemini V0 primary + user analysis |
| `M-P_llm_tests/results/GEMINI_3.1_PRO_FORENSIC_RESPONSE_v0.txt` | Gemini V0 forensic re-run (same window) |
| `M-P_llm_tests/results/GEMINI_3.1_PRO_PREVIEW_RESULTS_PROMPT_V1.txt` | Gemini V1 primary |
| `M-P_llm_tests/results/GEMINI_3.1_PRO_FORENSIC_RESPONSE_v1.txt` | Gemini V1 forensic re-run (same window) |
| `M-P_llm_tests/results/DEEPSEEK_PREVIEW_RESULTS_v0.txt` | DeepSeek V0 primary + user analysis |
| `M-P_llm_tests/results/DEEPSEEK_FORENSIC_RESPONSE_v0.txt` | DeepSeek V0 forensic re-run (same window) |
| `M-P_llm_tests/results/DEEPSEEK_PREVIEW_RESULTS_v1.txt` | DeepSeek V1 primary |
| `M-P_llm_tests/results/DEEPSEEK_FORENSIC_RESPONSE_v1.txt` | DeepSeek V1 forensic re-run (same window) |
| `M-P_llm_tests/ANALYSIS_SUMMARY.md` | Concise findings summary |
