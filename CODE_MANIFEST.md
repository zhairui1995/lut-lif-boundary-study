# Code Manifest

This repository is a standalone local extraction for the LUT spiking-neuron
boundary-study paper. It was copied from the sanitized anonymous artifact dry
run at `lut_if_paper/anonymous_artifact_dry_run_20260629`.

## Included

- `lut_if/`: Dense LUT-IF, CNL-LUT-LIF related replacement helpers, and
  quantized arithmetic LIF controls.
- `tools/`: reproduction entry points and QKFormer lookup probes used by the
  paper.
- `qkformer_lut/`: lookup primitives, hooks, cross-architecture hooks, and
  statistics helpers required by the included tools.
- `cifar10/`, `cifar100/`: QKFormer model definitions required by the tools.
- `run_specs/`: sanitized preregistered run specifications.
- `results/`: compact metrics, summary tables, and paper-facing reports.
- `figures/`: evidence-flow figure source and rendered outputs.
- `scripts/`: smoke/full command templates using environment variables.
- `docs/qklut_lif_architecture_pivot_20260630.md`: planned architecture pivot
  from boundary study to QK-LUT-LIF Transformer; no positive result is claimed
  by this planning note.
- `paper/`: anonymous manuscript PDF and reproducibility checklist draft.

## Excluded

- CIFAR datasets.
- Model checkpoints and model-weight files.
- Raw launcher logs and server/job-control logs.
- Private repository URLs, remote server addresses, usernames, tokens, or
  credentials.
- SDR-LUT/QK-LUTFormer paper materials that are unrelated to the standalone
  LUT spiking-neuron study.

## Repository Boundary

The copied files support a paper about controlled LUT replacement of LIF
neurons in QKFormer. They do not support claims about measured latency, energy,
area, SRAM, production acceleration, broad cross-architecture generalization,
seed stability, or bit-width robustness beyond the registered 8-bit diagnostic.

The added RL-guided policy probe now has a compact `NO-GO` result in
`results/rl_qklut_lif_policy_c100_t4_report.md`.

`tools/trainable_cnl_lut_lif_probe.py` adds the next planned architecture
probe: a trainable dense LUT-LIF transition module initialized with CNL current
normalization.

No GitHub remote is configured in this local repository.
