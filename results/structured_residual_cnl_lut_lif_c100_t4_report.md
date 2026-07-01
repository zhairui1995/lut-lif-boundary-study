# Structured Residual CNL-LUT-LIF Probe

Status: **NO-GO**

## Scope

- QKFormer cifar100, seed 42, T=4
- All discovered MultiStepLIF targets
- Fixed 6-bit state/input transition grid
- Frozen backbone; trainable threshold plus state/input/global structured residual over a fixed CNL-LUT-LIF table

## Results

| Method | Acc@1 | Drop | Logit MSE/sample |
|---|---:|---:|---:|
| posthoc_dense_transition_LUT | 78.9600 | 2.0600 | 24.577839 |
| quantized_arithmetic_LIF | 78.9000 | 2.1200 | 24.559828 |
| fixed_CNL_LUT_LIF | 78.9800 | 2.0400 | 24.570013 |
| structured_residual_CNL_LUT_LIF_initial | 78.7700 | 2.2500 | 22.710421 |
| structured_residual_CNL_LUT_LIF_final | 78.8600 | 2.1600 | 89.575663 |

## Gate

- Beats fixed CNL on Acc@1 or logit MSE: **False**
- Drop below fixed CNL: **False**
- Drop within threshold: **False**
- All requested targets executed: **True**

## Claim Boundary

This run tests whether a low-degree structured residual can avoid the dense
table drift observed in trainable CNL-LUT-LIF. It does not establish broad
architecture transfer, LL-ViT/QKFormer-family generalization, near-SOTA
accuracy, or hardware efficiency unless the registered full gate passes and is
followed by cross-setting evidence.
