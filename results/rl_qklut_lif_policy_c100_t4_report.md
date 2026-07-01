# RL-Guided QK-LUT-LIF Policy Probe

Status: **NO-GO**

## Scope

- QKFormer cifar100, seed 42, T=4
- All discovered MultiStepLIF targets grouped into 35 policy groups
- Actions per group: `posthoc`, `quant`, `cnl`
- Fixed coordinate-bandit budget: 1 round(s), no seed/checkpoint/bit-width search

## Search-Subset Fixed-Policy Controls

These rows use the same subset as the policy search and are not used for the
formal full-validation gate.

| Policy | Acc@1 | Drop | Logit MSE/sample | Reward |
|---|---:|---:|---:|---:|
| all_posthoc | 82.4219 | -1.5625 | 17.401283 | 1.388487 |
| all_quant | 82.0312 | -1.1719 | 17.444197 | 0.997433 |
| all_cnl | 82.4219 | -1.5625 | 17.390192 | 1.388598 |

## Full-Evaluation Fixed-Policy Controls

| Policy | Acc@1 | Drop | Logit MSE/sample | Reward |
|---|---:|---:|---:|---:|
| full_all_posthoc | 79.5500 | 1.4700 | 18.355140 | -1.653551 |
| full_all_quant | 79.6100 | 1.4100 | 18.291697 | -1.592917 |
| full_all_cnl | 79.7700 | 1.2500 | 18.350349 | -1.433503 |

## Learned Policy Result

- Acc@1: **79.5600%**
- Drop: **1.4600 pp**
- Logit MSE/sample: **18.353824**
- Reward: **-1.643538**

## Gate

- Beats best fixed control by reward: **False**
- Drop is within threshold: **True**
- All selected replacement modules executed: **True**

## Claim Boundary

This run tests a restricted policy-search mechanism for selecting LUT-LIF
replacement variants inside QKFormer. It can motivate a QK-LUT-LIF Transformer
architecture only if the policy improves the matched controls under the fixed
budget. It does not establish broad transfer, near-SOTA accuracy, hardware
efficiency, or LL-ViT compatibility.
