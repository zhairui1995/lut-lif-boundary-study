# QK-LUT-LIF Architecture Pivot

Date: 2026-07-01

Status: pivot under evaluation, with three completed `NO-GO` architecture
probes. The RL selector, dense trainable CNL table, and structured residual
CNL-LUT-LIF routes all fail their registered full-validation gates. No positive
new QK-LUT-LIF architecture result is claimed here.

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

Three cheap architecture pivots have now been tested and rejected:

1. **Selector policy over fixed variants.** Each LIF group chose one action:
   post-hoc transition LUT, quantized arithmetic LIF, or CNL-LUT-LIF. Result:
   `NO-GO`; the learned policy did not beat the fixed all-CNL-LUT-LIF
   full-validation control on QKFormer CIFAR-100 `T=4`. See
   `lut_if_paper/rl_qklut_lif_policy_result_20260701.md`.
2. **Dense trainable CNL transition table.** The backbone was frozen while
   dense transition tables, thresholds, and CNL affine parameters were trained.
   Result: `NO-GO`; training accuracy saturated while full-validation fidelity
   degraded, consistent with dense table drift under hard-address evaluation.
   See `lut_if_paper/trainable_cnl_lut_lif_result_20260701.md`.
3. **Structured residual CNL-LUT-LIF.** The analytic/CNL transition table was
   frozen and only state-vector, input-vector, global residual, and threshold
   parameters were trained. Result: `NO-GO`; final accuracy remained below the
   fixed CNL control and logit MSE degraded sharply. See
   `lut_if_paper/structured_residual_cnl_lut_lif_result_20260701.md`.

The remaining viable hypothesis is narrower: a positive QK-LUT-LIF architecture
will likely require native insertion and from-start or joint training, or a
different backbone interface, rather than post-hoc frozen-backbone replacement
with a small trainable correction.

## Experimental Order

1. QKFormer CIFAR-100 `T=4`, seed 42, smoke policy probe. `DONE`.
2. QKFormer CIFAR-100 `T=4`, seed 42, full policy probe. `NO-GO`.
3. Trainable dense CNL-LUT-LIF module: modify the module or training objective
   rather than selecting among fixed replacement variants. `NO-GO`.
4. Structured residual CNL-LUT-LIF module on QKFormer CIFAR-100 `T=4`,
   seed 42. `NO-GO`.
5. Cross-architecture smoke using existing Spikformer or QK-contract
   Spikformer support only if it is explicitly framed as a failure-boundary
   test or as a new preregistered native-training route.
6. LL-ViT adaptation only after locating a runnable implementation and mapping
   its LUT-based channel-mixer interface to this paper's SNN/LUT-LIF setting.

## Claim Boundary

Allowed only after a future positive native-training or cross-architecture run:

- a QK-LUT-LIF module improves a matched backbone under a registered protocol;
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
- CNL initialization plus dense trainable transition tables is sufficient to
  rescue LUT-LIF training. The completed full run contradicts this claim; see
  `lut_if_paper/trainable_cnl_lut_lif_result_20260701.md`.
- Structured residual CNL-LUT-LIF improves the QKFormer setting. The completed
  full run contradicts this claim; see
  `lut_if_paper/structured_residual_cnl_lut_lif_result_20260701.md`.

## Updated Route Decision

The evidence now rejects three cheap architecture pivots:

1. choosing among fixed replacement variants by a restricted policy;
2. training dense CNL-initialized transition tables after freezing the backbone;
3. training a low-degree residual over a frozen CNL transition table after
   freezing the backbone.

The structured residual attempt changed the parameterization but still failed
the registered full gate. The code is public on branch
`codex/qklut-lif-rl-architecture` at commit `0f5c9a3`; the full-run artifacts
are recorded under
`results/structured_residual_cnl_lut_lif_cifar100_t4_seed42_full_20260701_124750/`.

The route should not be rescued by a hidden sweep. The next defensible options
are either (i) from-start or joint training with QK-LUT-LIF inserted into the
backbone, (ii) an explicitly labeled cross-architecture boundary test on
existing QK-contract Spikformer support, or (iii) locating a runnable LL-ViT
implementation and preregistering a protocol-matched native-module test.
