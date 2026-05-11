# LLM Test Analysis Summary

**Date:** 2026-05-10  
**Benchmark:** Morgenstern-Price, 5 slices, constant f(x)=1, dry slope  
**Reference:** FS = 1.9710, λ = 0.2260, 4 iterations

---

## Results at a Glance

| Model | Prompt | FS | λ | E_n | Verdict |
|---|---|---|---|---|---|
| Gemini  | V0 | 1.965 | 0.212 | 0.00 | Near-correct FS, λ wrong |
| Gemini  | V1 | 1.881 | 0.053 | 0.00 | Both wrong, degeneracy visible |
| DeepSeek | V0 | 1.886 | −0.142 | 0.00 | α inverted, λ negative |
| DeepSeek | V1 | 1.542 | 0.234 | −48.2 | α inverted, fsolve admitted |

---

## Key Findings

### 1. Universal Ψ Degeneracy (Mode 3)

**Every model tested produced Ψ = 1 for constant f(x) = 1.**

The correct Zhu et al. formula uses mixed indexing: numerator depends on slice j+1, denominator
depends on Φ_j from the current slice. Both tested models implemented a formula where numerator
and denominator reference the same slice, so Ψ collapses to 1 when f(x) is constant.

This is the single most important finding: the structural error in Ψ is universal to all
tested models, yet it was invisible in Gemini V0 because the near-correct FS masked it.

### 2. DeepSeek α Sign Inversion (Mode 1)

DeepSeek used `α = arctan((x_mid − x_c)/(y_c − y_bot))`, producing the inverted sign
convention (α < 0 in active zone). This manifests as:
- V0: λ = −0.142 (negative, physically unusual)
- V1: FS = 1.542 (far from reference, boundary condition violated)

### 3. DeepSeek fsolve Admission (Mode 2)

In V0, DeepSeek described its solver as "secant method on the force residual" — ambiguous.
The V1 prompt's explicit B2 question elicited: "solved using a two-variable secant/Newton
solver (equivalent to scipy.optimize.fsolve)." This explains the E_n ≠ 0 in V1.

### 4. V1 Prompt Made Failures MORE Visible (Not Less)

Both models produced numerically worse results with V1 — but the failures became structurally
explicit:
- Gemini V1 B4: Ψ = 1.000 (explicitly stated, not inferred)
- DeepSeek V1 B2: fsolve admitted (V0 had hidden this)
- DeepSeek V1: E_n = −48.2 kN reported (V0 showed E_n = 0)

This confirms the design philosophy: clean benchmark + forensic questionnaire > prescriptive hint-laden prompt.

### 5. Forensic Re-Run Did Not Trigger Self-Correction

**Note:** In this experiment, the "forensic" run was a re-application of the primary prompt
(`FORENSIC_QUESTIONAIRE.md` = same content as primary prompt) in the same conversation window,
not the structured F1–F7 diagnostic questionnaire.

Both models reproduced their original solutions exactly. The structured F1–F7 questionnaire
has not been empirically tested yet. The V1 prompt improvements (B4 explicit Ψ evaluation,
explicit B2 solver question) effectively served as embedded forensic probes within the primary prompt.

---

## Unresolved Questions

1. **Mode 3 with non-constant f(x):** Would the half_sine function expose Ψ errors that
   f(x)=1 masks? Expected: yes, because Ψ ≠ 1 only when f differs between boundaries.
2. **Gemini V0 near-correct FS:** Is FS = 1.965 a geometric coincidence of this benchmark,
   or would the same formula consistently produce ~0.3% error across different geometries?
3. **Forensic in fresh session:** Would models self-correct if the forensic questionnaire
   were posed in a fresh context (without the prior primary prompt contaminating the context)?

---

## Recommended Next Tests

1. Run both models with half_sine f(x) to expose Mode 5 (f_left/f_right indexing)
2. Apply forensic questionnaire in a **fresh conversation window** to test self-correction
3. Test with a steeper slope or circular geometry to see if Mode 3 FS deviation scales up
4. Test with a model that uses a different tokenization of mathematical notation (GPT-4o)
