# Trainable CNL-LUT-LIF Probe

Status: **NO-GO**

## Scope

- QKFormer cifar100, seed 42, T=4
- All discovered MultiStepLIF targets
- Fixed 6-bit state/input dense trainable transition LUT
- Frozen backbone; trainable transition tables, thresholds, and CNL affine parameters

## Results

| Method | Acc@1 | Drop | Logit MSE/sample |
|---|---:|---:|---:|
| posthoc_dense_transition_LUT | 78.9600 | 2.0600 | 24.577839 |
| quantized_arithmetic_LIF | 78.9000 | 2.1200 | 24.559828 |
| fixed_CNL_LUT_LIF | 78.9800 | 2.0400 | 24.570013 |
| trainable_CNL_LUT_LIF_initial | 78.7700 | 2.2500 | 22.710421 |
| trainable_CNL_LUT_LIF_final | 78.5800 | 2.4400 | 82.543650 |

## Gate

- Beats fixed CNL on Acc@1 or logit MSE: **False**
- Drop below fixed CNL: **False**
- Drop within threshold: **False**
- All requested targets executed: **True**

## Claim Boundary

This run tests whether QK-LUT-inspired current normalization can rescue the
trainable LUT-LIF objective. It does not establish broad architecture transfer,
near-SOTA accuracy, or hardware efficiency unless the fixed full-validation
gate passes and is followed by cross-setting evidence.
