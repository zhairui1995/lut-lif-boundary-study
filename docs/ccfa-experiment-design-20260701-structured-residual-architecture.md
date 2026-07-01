# CCFA Experiment Design: Structured Residual QK-LUT-LIF Architecture Probe

Date: 2026-07-01

Mode: experiment design and completed-result record.

## Venue And Assumptions

- Target: AAAI-style CCF-A main-track evidence package.
- Paper identity under test: a QK-LUT-inspired spiking Transformer module, not
  a post-hoc lookup replacement audit.
- Current evidence: RL selector, dense trainable CNL-LUT-LIF, and structured
  residual CNL-LUT-LIF are completed `NO-GO` results. Fixed CNL-LUT-LIF remains
  the strongest control in the evaluated QKFormer CIFAR-100 `T=4` setting.
- No near-SOTA, broad-transfer, LL-ViT, hardware, latency, energy, or SRAM
  claim is allowed before matched evidence exists.

## Claim-Evidence Matrix

| Claim | Reviewer question | Evidence needed | Dataset/benchmark | Baselines | Metrics | Result placeholder | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Structured residual CNL-LUT-LIF is a stronger module than fixed CNL-LUT-LIF on the seed-42 QKFormer setting. | Does the proposed trainable module improve over the best fixed lookup control? | Registered full run with same checkpoint, calibration budget, full validation, and all LIF targets executed. | CIFAR-100, QKFormer, `T=4`, seed 42. | LIF teacher, posthoc dense transition LUT, quantized arithmetic LIF, fixed CNL-LUT-LIF, structured residual initialization. | Acc@1, paired drop, logit MSE/sample, coverage. | Final 78.86 vs fixed CNL 78.98; drop 2.16 vs 2.04; logit MSE 89.5757 vs 24.5700. | done / NO-GO |
| Low-degree residual avoids dense table drift. | Why should this trainable route succeed where dense CNL failed? | Compare train/validation behavior and final full-validation gate against dense CNL `NO-GO`. | Same as above. | Dense trainable CNL-LUT-LIF `NO-GO`, fixed CNL control. | train_top1, val Acc@1, logit MSE. | train_top1 reaches 99.71%--100.00%, while final full-validation logit MSE worsens. | done / NO-GO |
| QK-LUT principles motivate an architecture, not only an insertion policy. | Is this just selecting existing modules? | Show the active trainable degrees of freedom are structured residual parameters over a fixed CNL table, not a policy over fixed variants. | Same as above. | RL selector `NO-GO`; fixed all-CNL. | gate verdict plus trainable parameter accounting. | The parameterization is distinct from selection, but the result is still `NO-GO`. | done / negative |
| Cross-architecture compatibility is plausible. | Does it transfer beyond QKFormer? | Only run a new preregistered native-training or boundary test; do not trigger it from this failed gate as a positive-claim extension. | TBD. | Original target backbone, fixed CNL/quant/posthoc analogues if compatible. | Acc@1/drop/coverage. | TBD | blocked by failed QKFormer gate |

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

Completed full result:

| Method | Acc@1 | Drop | Logit MSE/sample | Gate |
| --- | ---: | ---: | ---: | --- |
| Posthoc dense transition LUT | 78.96 | 2.06 | 24.577839 | control |
| Quantized arithmetic LIF | 78.90 | 2.12 | 24.559828 | control |
| Fixed CNL-LUT-LIF | 78.98 | 2.04 | 24.570013 | best fixed control |
| Structured residual CNL-LUT-LIF init | 78.77 | 2.25 | 22.710421 | init |
| Structured residual CNL-LUT-LIF final | 78.86 | 2.16 | 89.575663 | `NO-GO` |

All 35 requested targets executed, but the structured residual final row did
not beat fixed CNL on Acc@1 or logit MSE, did not reduce the fixed-CNL drop,
and exceeded the 1.25 pp drop threshold.

## Ablations

| Variant | Component changed | Mechanism tested | Metric | Interpretation after result |
| --- | --- | --- | --- | --- |
| Structured residual full | state + input + global residual | Whether low-degree correction helps without dense drift. | Acc@1/drop/logit MSE/residual RMS | Main gate. |
| Dense trainable CNL | dense table entries | Whether unrestricted table learning overfits. | Existing Acc@1/drop/logit MSE | Already failed; supports need for structure. |
| Fixed CNL | no trainable residual | Whether training adds value. | Acc@1/drop/logit MSE | Must be beaten. |

Do not add new ablations until the main gate finishes. Additional variants
require a new run spec.

## Cross-Architecture Queue

Priority order after the failed QKFormer gate:

1. Record structured residual as a boundary result; do not sweep the same
   post-hoc frozen-backbone setting.
2. Consider from-start or joint training where QK-LUT-LIF is inside the
   backbone from epoch 0.
3. A minimal QK-contract Spikformer compatibility test may be run only as a
   preregistered failure-boundary test or native-training route, not as a
   hidden rescue.
4. LL-ViT is not currently runnable in this repository. It remains the closest
   novelty-risk comparator and an adaptation target only after a public
   implementation/checkpoint path is located and mapped to the SNN/LUT-LIF
   interface.

## Result Status

Smoke passed and the registered full run completed on an anonymous GPU server.
The formal verdict is `NO-GO`. Evidence note:
`lut_if_paper/structured_residual_cnl_lut_lif_result_20260701.md`.

## No-Fabrication Status

This memo now records real values from the registered server run. Remaining
`TBD` values are future cross-architecture placeholders only. This memo does
not support near-SOTA, broad architecture transfer, or LL-ViT compatibility
claims.
