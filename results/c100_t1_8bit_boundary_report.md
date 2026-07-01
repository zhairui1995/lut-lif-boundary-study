# CNL-LUT-LIF CIFAR-100 T=1 8-bit Boundary Diagnostic

Run date: 2026-06-29

Run spec:
`run_specs/cnl_lut_lif_c100_t1_bit8_boundary_seed42_20260629.yaml`

Remote result:
`<anonymous-gpu-node>/results/cnl_lut_lif_c100_t1_bit8_boundary_seed42_full_20260629_083155`

Local artifact copy:
`results/cnl_lut_lif_c100_t1_bit8_boundary_seed42_full_20260629_083155/`

## Verdict

Primary diagnostic gate: **GO**.

Secondary CNL method gate: **NO-GO**.

Interpretation: raising only the fixed LIF transition state/input precision
from 6-bit to 8-bit reduces the CIFAR-100 `T=1` all-LIF replacement drop from
`6.79` pp to `0.46` pp for the post-hoc/CNL rows. This supports a
setting-specific precision/state-range boundary diagnosis for the severe
6-bit CIFAR-100 `T=1` failure. It does not support a broader claim that
CNL-LUT-LIF is superior to same-run post-hoc dense transition lookup, because
CNL ties post-hoc at 8-bit.

## Protocol

| Field | Value |
| --- | --- |
| Dataset | CIFAR-100 |
| Model | QKFormer retained checkpoint |
| Time steps | `T=1` |
| Seed | 42 |
| Calibration | 128 batches |
| Evaluation | 10,000 validation samples |
| LIF targets | 35 / 35 executed |
| State bits | 8 |
| Input bits | 8 |
| Code snapshot | remote branch `codex/cnl-lut-lif-e0-run`, commit `959782f` |

The raw tool report keeps legacy method names containing `6bit`; the protocol
field in `metrics.json` records `state_bits=8` and `input_bits=8`, and this
summary labels the rows by the registered 8-bit protocol.

## Main Results

| Method | Acc@1 | Drop | Logit MSE/sample | R3 role |
| --- | ---: | ---: | ---: | --- |
| 6-bit post-hoc dense LUT, prior matrix | 70.79 | 6.79 | 49.3980 | fixed prior failure row |
| 6-bit CNL normalized, prior matrix | 70.79 | 6.79 | 49.3980 | fixed prior failure row |
| 8-bit post-hoc dense LUT | 77.12 | 0.46 | 21.1953 | primary diagnostic support |
| 8-bit quantized arithmetic LIF | 77.07 | 0.51 | 21.2273 | matched arithmetic control |
| 8-bit CNL without normalization | 77.12 | 0.46 | 21.1953 | normalization ablation |
| 8-bit CNL with normalization | 77.12 | 0.46 | 21.1953 | CNL candidate |

Teacher Acc@1 is `77.58`.

## Diagnostics

| Diagnostic | Value |
| --- | ---: |
| Current MSE before normalization | 0.0009421779 |
| Current MSE after normalization | 0.0009296279 |
| Post-hoc aggregate transition clip rate | 6.330353e-10 |
| CNL aggregate transition clip rate | 6.330353e-10 |
| Post-hoc table entries | 4,587,520 |
| Post-hoc FP32 value proxy | 17,920.0 KiB |
| CNL metadata-inclusive proxy | 18,107.0 KiB |

## Gate Evaluation

Primary diagnostic gate:

- Full validation completed: **pass**.
- All requested LIF targets executed: **pass**, 35 / 35.
- No nonfinite metrics: **pass**.
- 8-bit post-hoc or 8-bit CNL normalized drop `<= 4.79` pp: **pass**,
  observed `0.46` pp.

Secondary CNL method gate:

- CNL normalized beats same-run 8-bit post-hoc on paired drop or logit MSE:
  **fail**, tied at `0.46` pp drop and `21.1953` logit MSE/sample.
- CNL normalized paired drop is lower than same-run post-hoc: **fail**, tied.

## Claim Boundary

Allowed:

- In the evaluated QKFormer CIFAR-100 `T=1` seed-42 setting, the severe 6-bit
  all-LIF replacement failure is precision/state-range sensitive.
- The CIFAR-100 `T=1` row is an interpretable bit-width boundary diagnostic
  rather than an uninterpreted negative result.

Not allowed:

- CNL-LUT-LIF is broadly superior to post-hoc dense transition LUT.
- 8-bit is an optimal or generally robust precision.
- The result proves seed stability, cross-architecture transfer, hardware
  speed, energy, latency, SRAM, area, or deployment compactness.
