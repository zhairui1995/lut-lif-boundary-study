# CCFA Experiment Design: Structured Residual QK-LUT-LIF Architecture Probe

Date: 2026-07-01

Mode: experiment design and execution queue.

## Venue And Assumptions

- Target: AAAI-style CCF-A main-track evidence package.
- Paper identity under test: a QK-LUT-inspired spiking Transformer module, not
  a post-hoc lookup replacement audit.
- Current evidence: RL selector and dense trainable CNL-LUT-LIF are completed
  `NO-GO` results. Fixed CNL-LUT-LIF remains the strongest control in the
  evaluated QKFormer CIFAR-100 `T=4` setting.
- No near-SOTA, broad-transfer, LL-ViT, hardware, latency, energy, or SRAM
  claim is allowed before matched evidence exists.

## Claim-Evidence Matrix

| Claim | Reviewer question | Evidence needed | Dataset/benchmark | Baselines | Metrics | Result placeholder | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Structured residual CNL-LUT-LIF is a stronger module than fixed CNL-LUT-LIF on the seed-42 QKFormer setting. | Does the proposed trainable module improve over the best fixed lookup control? | Registered full run with same checkpoint, calibration budget, full validation, and all LIF targets executed. | CIFAR-100, QKFormer, `T=4`, seed 42. | LIF teacher, posthoc dense transition LUT, quantized arithmetic LIF, fixed CNL-LUT-LIF, structured residual initialization. | Acc@1, paired drop, logit MSE/sample, coverage, residual size. | TBD | planned/code pushed/server pending |
| Low-degree residual avoids dense table drift. | Why should this trainable route succeed where dense CNL failed? | Compare train/validation behavior, residual RMS, hard/soft lookup MSE, and final full-validation gate against dense CNL `NO-GO`. | Same as above. | Dense trainable CNL-LUT-LIF `NO-GO`, fixed CNL control. | train_top1, val Acc@1, logit MSE, structured_delta_rms, hard_soft_mse. | TBD | planned |
| QK-LUT principles motivate an architecture, not only an insertion policy. | Is this just selecting existing modules? | Show the active trainable degrees of freedom are structured residual parameters over a fixed CNL table, not a policy over fixed variants. | Same as above. | RL selector `NO-GO`; fixed all-CNL. | gate verdict plus trainable parameter accounting. | TBD | planned |
| Cross-architecture compatibility is plausible. | Does it transfer beyond QKFormer? | Only after QKFormer structured residual passes, run a preregistered smoke/full boundary test on QK-contract Spikformer or locate a runnable LL-ViT implementation. | TBD. | Original target backbone, fixed CNL/quant/posthoc analogues if compatible. | Acc@1/drop/coverage. | TBD | blocked by prior gate |

## Baseline Matrix

| Baseline | Why included | Implementation source | Fairness constraints | Expected metric | Can run? |
| --- | --- | --- | --- | --- | --- |
| Original frozen LIF teacher | Defines paired target behavior. | QKFormer checkpoint and `tools/lut_if_poc.py` loaders. | Same checkpoint, full validation. | Teacher Acc@1/logits. | yes, on server |
| Posthoc dense transition LUT | Strong direct finite-state lookup control. | `replace_lif_targets` from native affine/LIF probe. | Same ranges/calibration. | Acc@1/drop/logit MSE. | yes |
| Quantized arithmetic LIF | Tests whether lookup table itself adds value beyond matched quantization. | `lut_if.replace.replace_with_quantized_arithmetic`. | Same state/input bits. | Acc@1/drop/logit MSE. | yes |
| Fixed CNL-LUT-LIF | Strongest existing CNL control. | `tools/cnl_lut_lif_e0.py`. | Same moments/calibration/evaluation. | Acc@1/drop/logit MSE. | yes |
| Dense trainable CNL-LUT-LIF | Prior failed trainable route. | `tools/trainable_cnl_lut_lif_probe.py`. | Use as evidence note, not rerun. | NO-GO comparison. | already done |
| RL selector policy | Prior failed architecture route. | `tools/rl_lut_lif_policy_probe.py`. | Use as evidence note, not rerun. | NO-GO comparison. | already done |

## Main Experiment

Run spec:
`run_specs/structured_residual_cnl_lut_lif_qkformer_c100_t4_seed42_20260701.yaml`.

Command:
`scripts/server/run_structured_residual_cnl_lut_lif_probe.sh`.

Fixed full budget:

- CIFAR-100 QKFormer `T=4`, seed 42.
- 6-bit state, 6-bit input.
- 32 calibration batches.
- 3 epochs.
- 128 train batches per epoch.
- Complete 10,000-image validation.
- Frozen backbone.
- Trainable variables: threshold plus state-vector, input-vector, and global
  residual over a fixed analytic/CNL transition table.

Gate:

- Smoke must execute all requested replacement modules and complete one
  backward update.
- Full structured residual CNL-LUT-LIF must beat fixed all-CNL-LUT-LIF on
  Acc@1 or logit MSE.
- Full paired Acc@1 drop must be below the fixed CNL drop.
- Full paired Acc@1 drop must be `<= 1.25` percentage points.
- Stop after this registered run; no hidden learning-rate, bit-width, seed,
  checkpoint, residual-rank, architecture, or retry-until-positive search.

## Ablations

| Variant | Component changed | Mechanism tested | Metric | Interpretation after result |
| --- | --- | --- | --- | --- |
| Structured residual full | state + input + global residual | Whether low-degree correction helps without dense drift. | Acc@1/drop/logit MSE/residual RMS | Main gate. |
| Dense trainable CNL | dense table entries | Whether unrestricted table learning overfits. | Existing Acc@1/drop/logit MSE | Already failed; supports need for structure. |
| Fixed CNL | no trainable residual | Whether training adds value. | Acc@1/drop/logit MSE | Must be beaten. |

Do not add new ablations until the main gate finishes. Additional variants
require a new run spec.

## Cross-Architecture Queue

Priority order after QKFormer gate:

1. If structured residual passes: run a minimal QK-contract Spikformer
   compatibility test using existing Spikformer support. This tests whether
   the QK-style contract is a viable architecture substrate.
2. If structured residual fails but the failure is informative: write it as a
   boundary result and consider from-start training where QK-LUT-LIF is inside
   the backbone from epoch 0.
3. LL-ViT is not currently runnable in this repository. It remains the closest
   novelty-risk comparator and an adaptation target only after a public
   implementation/checkpoint path is located and mapped to the SNN/LUT-LIF
   interface.

## Current Blocker

Remote execution is pending because the configured private remote server is
not reachable from the local machine. No smoke/full result has been generated.

## No-Fabrication Status

No experimental result is generated in this memo. All `TBD` values must be
filled only from the registered server run or from already recorded evidence
notes. This memo does not support near-SOTA, broad architecture transfer, or
LL-ViT compatibility claims.
