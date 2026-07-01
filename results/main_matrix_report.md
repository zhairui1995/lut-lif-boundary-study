# CNL-LUT-LIF Main Comparison Matrix

Matrix gate: 1/4 per-run gates passed.

| Run | Teacher | Post-hoc LUT | Quantized LIF | CNL no norm | CNL norm | Gate |
|---|---:|---:|---:|---:|---:|---|
| cifar100 T=1 | 77.58 | 70.79 (6.79) | 70.87 (6.71) | 70.79 (6.79) | 70.79 (6.79) | NO-GO |
| cifar100 T=4 | 81.02 | 79.55 (1.47) | 79.61 (1.41) | 79.55 (1.47) | 79.77 (1.25) | GO |
| cifar10 T=1 | 94.90 | 94.53 (0.37) | 94.43 (0.47) | 94.53 (0.37) | 94.53 (0.37) | NO-GO |
| cifar10 T=4 | 95.73 | 95.64 (0.09) | 95.54 (0.19) | 95.64 (0.09) | 95.64 (0.09) | NO-GO |

## Gate Details

- cifar100 T=1: verdict=NO-GO, all_targets=True, beats_posthoc=False, drop_below_same_posthoc=False, current_mse_improves=True, current MSE after=0.0144191.
- cifar100 T=4: verdict=GO, all_targets=True, beats_posthoc=True, drop_below_same_posthoc=True, current_mse_improves=True, current MSE after=0.0246906.
- cifar10 T=1: verdict=NO-GO, all_targets=True, beats_posthoc=True, drop_below_same_posthoc=False, current_mse_improves=True, current MSE after=0.0148952.
- cifar10 T=4: verdict=NO-GO, all_targets=True, beats_posthoc=False, drop_below_same_posthoc=False, current_mse_improves=True, current MSE after=0.023026.
