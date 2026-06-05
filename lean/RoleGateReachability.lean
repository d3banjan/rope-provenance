import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic

/-!
# Role-gate monotone reachability (RPCG3 co-evolution)

Descriptive co-evolution theorem for the role-provenance capability-gate
experiment (`preregistry/rpcg3_dpo_explicit_negatives_pythia70m_2026-05-18/`).

RPCG3 (DPO + chosen-NLL anchor on Pythia-70m) returned `GATE_INSTALLED`:
alignment-style contrastive tuning installs a role-conditioned gate that
*suppresses* the output logits of primitives outside the active role's
permission set, and the gate lives in a low-rank update to the attention QKV
projection (measured stable rank ≈ 1.30).

This file formalises what the experiment's *ablation* readout would observe.
Model the gated logit of one out-of-role primitive as a base (un-gated) logit
plus a finite family of rank-1 gate components, each a **suppression**
(`comp i ≤ 0`). "Ablating a rank-`r` subspace of the gate update" = deleting
`r` of those components — the surviving gate contribution is the sum over the
components that were *not* ablated.

The theorems below state **monotone reachability under low-rank gate removal**:

* `logit_mono` — ablating *more* components never lowers the logit; it
  monotonically rises.
* `logit_full_ablation` — ablating the whole gate reaches exactly the un-gated
  `base` logit.
* `logit_le_base` — under any partial ablation the logit stays at or below the
  un-gated value (the gate only ever suppresses).
* `ablation_monotonically_reaches_base` — the capstone: the two facts hold
  simultaneously for *every* out-of-role primitive.

Descriptive only. No minimality (least-privilege optimality) is claimed — that
is deliberately deferred, matching the v1 experiment scope.
-/

namespace RoleProvenance
namespace RoleGate

open Finset

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The gated logit of a single out-of-role primitive. `base` is the un-gated
logit; the role gate is a finite family of rank-1 components indexed by `ι`,
each contributing `comp i`. `Fintype.card ι` is the rank of the gate update. -/
structure GatedLogit (ι : Type*) where
  /-- the un-gated (gate-free) logit of the primitive -/
  base : ℝ
  /-- contribution of each rank-1 gate component -/
  comp : ι → ℝ

namespace GatedLogit

/-- The logit after ablating the gate components in `S`. The ablated components
are removed; the gate's surviving contribution is the sum over the complement. -/
def logit (G : GatedLogit ι) (S : Finset ι) : ℝ :=
  G.base + ∑ i ∈ Sᶜ, G.comp i

/-- A gate is *suppressing* when every rank-1 component lowers the logit — the
empirical signature of a role gate acting on an out-of-role primitive. -/
def Suppressing (G : GatedLogit ι) : Prop := ∀ i, G.comp i ≤ 0

/-- Ablating the entire gate (`S = univ`) reaches exactly the un-gated logit. -/
theorem logit_full_ablation (G : GatedLogit ι) :
    G.logit Finset.univ = G.base := by
  simp [logit]

/-- **Monotone reachability.** Ablating more gate components (`S ⊆ T`) never
lowers the logit of a suppressing gate — it monotonically rises. -/
theorem logit_mono (G : GatedLogit ι) (hG : G.Suppressing)
    {S T : Finset ι} (hST : S ⊆ T) : G.logit S ≤ G.logit T := by
  have hc : Tᶜ ⊆ Sᶜ := Finset.compl_subset_compl.mpr hST
  have hsum : ∑ i ∈ Sᶜ, G.comp i ≤ ∑ i ∈ Tᶜ, G.comp i :=
    Finset.sum_le_sum_of_subset_of_nonpos hc (fun i _ _ => hG i)
  unfold logit
  linarith

/-- The full gate suppresses: with no ablation the logit sits at or below the
un-gated `base` value. -/
theorem logit_no_ablation_le_base (G : GatedLogit ι) (hG : G.Suppressing) :
    G.logit ∅ ≤ G.base := by
  have hsum : ∑ i ∈ (∅ : Finset ι)ᶜ, G.comp i ≤ 0 :=
    Finset.sum_nonpos (fun i _ => hG i)
  unfold logit
  linarith

/-- Under *any* partial ablation the logit of a suppressing gate stays at or
below the un-gated value — the gate only ever suppresses. -/
theorem logit_le_base (G : GatedLogit ι) (hG : G.Suppressing) (S : Finset ι) :
    G.logit S ≤ G.base := by
  have h := logit_mono G hG (Finset.subset_univ S)
  rwa [logit_full_ablation] at h

end GatedLogit

/-- **Capstone — monotone reachability under low-rank gate removal.**

For a family of out-of-role primitives, each carrying a suppressing role gate:
ablating more gate components (`S ⊆ T`) monotonically raises *every* primitive's
logit, and ablating the whole gate reaches *every* primitive's un-gated value.

This is the descriptive shape the RPCG3 ablation readout is predicted to trace:
walking the out-of-role logits monotonically from their suppressed (full-gate)
values back up to the un-gated baseline as the low-rank gate is removed. -/
theorem ablation_monotonically_reaches_base
    {P : Type*} (G : P → GatedLogit ι)
    (hsupp : ∀ p, (G p).Suppressing)
    {S T : Finset ι} (hST : S ⊆ T) :
    (∀ p, (G p).logit S ≤ (G p).logit T) ∧
    (∀ p, (G p).logit Finset.univ = (G p).base) :=
  ⟨fun p => GatedLogit.logit_mono (G p) (hsupp p) hST,
   fun p => GatedLogit.logit_full_ablation (G p)⟩

end RoleGate
end RoleProvenance
