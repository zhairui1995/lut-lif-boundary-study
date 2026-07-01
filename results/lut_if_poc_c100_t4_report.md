# LUT-IF Initial Proof-of-Concept

Status: **NO-GO**

## Scope

- QKFormer CIFAR-100, seed 42, T=4
- LIF-only intervention
- 6-bit state and 6-bit input addresses
- Frozen backbone; only LUT-IF transition tables and thresholds are trained

## Baselines Before Training

| Method | Acc@1 | Drop | Logit MSE/sample |
|---|---:|---:|---:|
| Post-hoc dense LUT | 78.9600 | 2.0600 | 24.577839 |
| Quantized arithmetic LIF | 78.9000 | 2.1200 | 24.559828 |
| Trainable LUT-IF initialization | 78.7700 | 2.2500 | 24.547122 |

## Final LUT-IF Result

- Acc@1: **78.4200%**
- Paired drop: **2.6000 pp**
- Logit MSE/sample: **56.616155**
- Initial proof-of-concept gate: **NO-GO**

## Claim Boundary

This run can only test whether a trainable LUT-IF module provides an initial matched-precision advantage on the registered QKFormer setting. It does not establish compact factorization, broad transfer, or hardware efficiency.
