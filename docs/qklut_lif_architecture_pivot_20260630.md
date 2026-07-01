# QK-LUT-LIF Architecture Pivot

Date: 2026-06-30

Status: pivot under evaluation; no new positive architecture result is claimed
here.

## New Paper Identity

The third paper should pivot from a pure boundary study to a proposed spiking
Transformer architecture:

> QK-LUT-LIF Transformer: a spiking Transformer family that inserts
> lookup-parameterized LIF transition modules according to QK-LUT-inspired
> semantic addressability, support-aware selection, and temporal/channel
> current normalization.

The intended contribution is a new network/module design, not only a
post-hoc replacement audit. The paper may pursue near-SOTA accuracy only after
matched experiments verify the claim.

## Current Hypothesis

Second-paper QK-LUT findings suggest three design principles:

1. addresses should be semantically structured rather than arbitrary bins;
2. unsupported addresses must be exposed and controlled, not hidden;
3. temporal/channel current normalization can reduce local mismatch in some
   QKFormer settings.

The first falsifiable experiment treated insertion as a restricted policy
problem. Each LIF group chose one action: post-hoc transition LUT, quantized
arithmetic LIF, or CNL-LUT-LIF. A fixed coordinate-bandit pass selected the
policy under a smoke/full budget.

Result: this selector-style route is `NO-GO`. The learned policy did not beat
the fixed all-CNL-LUT-LIF full-validation control on QKFormer CIFAR-100 `T=4`.
See `lut_if_paper/rl_qklut_lif_policy_result_20260701.md`.

## Experimental Order

1. QKFormer CIFAR-100 `T=4`, seed 42, smoke policy probe. `DONE`.
2. QKFormer CIFAR-100 `T=4`, seed 42, full policy probe. `NO-GO`.
3. Trainable QK-LUT-LIF module: modify the module or training objective rather
   than selecting among fixed replacement variants. `NEXT`.
4. Cross-architecture smoke using existing Spikformer support if compatible.
5. LL-ViT adaptation only after locating a runnable implementation and mapping
   its LUT-based channel-mixer interface to this paper's SNN/LUT-LIF setting.

## Claim Boundary

Allowed after a future passing full QKFormer run:

- a policy-guided LUT-LIF insertion can improve over fixed insertion policies
  on the evaluated QKFormer setting;
- QK-LUT principles motivate the architecture design.

Not allowed without additional evidence:

- near-SOTA accuracy;
- broad backbone transfer;
- LL-ViT compatibility;
- hardware speed, latency, energy, SRAM, or area;
- superiority over Spikformer/Spikeformer/LL-ViT.

Current disallowed claim:

- RL/bandit selection over fixed post-hoc, quantized, and CNL replacement
  variants improves the QKFormer setting. The completed full run contradicts
  this claim.
