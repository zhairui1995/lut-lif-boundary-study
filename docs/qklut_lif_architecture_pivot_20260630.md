# QK-LUT-LIF Architecture Pivot

Date: 2026-07-01

Status: pivot under evaluation. The RL selector and dense trainable CNL table
routes are completed `NO-GO` results. The active architecture gate is the
registered structured residual CNL-LUT-LIF probe; no new positive architecture
result is claimed here.

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

Two cheap architecture pivots have now been tested and rejected:

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

The active hypothesis is narrower and more causal: fixed CNL-LUT-LIF may be
useful, but the trainable part must be low-degree and structured. The current
registered probe therefore freezes the analytic/CNL transition table and
trains only a state-vector residual, input-vector residual, global residual,
and threshold while preserving hard-address evaluation.

## Experimental Order

1. QKFormer CIFAR-100 `T=4`, seed 42, smoke policy probe. `DONE`.
2. QKFormer CIFAR-100 `T=4`, seed 42, full policy probe. `NO-GO`.
3. Trainable dense CNL-LUT-LIF module: modify the module or training objective
   rather than selecting among fixed replacement variants. `NO-GO`.
4. Structured residual CNL-LUT-LIF module on QKFormer CIFAR-100 `T=4`,
   seed 42. `PLANNED / CODE_PUSHED / SERVER_PENDING`.
5. Cross-architecture smoke using existing Spikformer or QK-contract
   Spikformer support only if step 4 passes or if it is explicitly framed as a
   failure-boundary test.
6. LL-ViT adaptation only after locating a runnable implementation and mapping
   its LUT-based channel-mixer interface to this paper's SNN/LUT-LIF setting.

## Claim Boundary

Allowed after a future passing full QKFormer structured-residual run:

- a structured residual LUT-LIF module can improve over fixed CNL-LUT-LIF on
  the evaluated QKFormer setting;
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
- Structured residual CNL-LUT-LIF improves the QKFormer setting. This remains
  a planned claim until the registered full run completes and passes.

## Updated Route Decision

The evidence now rejects both cheap architecture pivots:

1. choosing among fixed replacement variants by a restricted policy;
2. training dense CNL-initialized transition tables after freezing the backbone.

The next architecture attempt changes the parameterization, not the selection
rule. The structured residual module has been implemented and preregistered in
`run_specs/structured_residual_cnl_lut_lif_qkformer_c100_t4_seed42_20260701.yaml`.
The code is public on branch `codex/qklut-lif-rl-architecture` at commit
`86519d9`; the standalone public extraction is updated at `d22f1c8`.

Execution is currently pending because the configured private remote server is
not reachable from the local machine. Do not report smoke/full results until
the registered runner executes on the server.

If the structured residual gate passes, the paper can continue toward a new
module-architecture claim. If it fails, the route should not be rescued by a
hidden sweep; the next defensible options are either (i) from-start training
with QK-LUT-LIF inserted into the backbone or (ii) an explicitly labeled
cross-architecture boundary test on the existing QK-contract Spikformer
support.
