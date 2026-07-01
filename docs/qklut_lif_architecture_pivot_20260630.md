# QK-LUT-LIF Architecture Pivot

Date: 2026-06-30

Status: planned pivot; no new positive result is claimed here.

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

The first falsifiable experiment treats insertion as a restricted policy
problem. Each LIF group chooses one action: post-hoc transition LUT, quantized
arithmetic LIF, or CNL-LUT-LIF. A fixed coordinate-bandit pass selects the
policy under a smoke/full budget.

## Experimental Order

1. QKFormer CIFAR-100 `T=4`, seed 42, smoke policy probe.
2. If smoke passes, QKFormer CIFAR-100 `T=4`, seed 42, full policy probe.
3. Cross-architecture smoke using existing Spikformer support if compatible.
4. LL-ViT adaptation only after locating a runnable implementation and mapping
   its LUT-based channel-mixer interface to this paper's SNN/LUT-LIF setting.

## Claim Boundary

Allowed after a passing full QKFormer run:

- a policy-guided LUT-LIF insertion can improve over fixed insertion policies
  on the evaluated QKFormer setting;
- QK-LUT principles motivate the architecture design.

Not allowed without additional evidence:

- near-SOTA accuracy;
- broad backbone transfer;
- LL-ViT compatibility;
- hardware speed, latency, energy, SRAM, or area;
- superiority over Spikformer/Spikeformer/LL-ViT.
