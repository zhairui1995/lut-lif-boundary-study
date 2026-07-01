# LUT Spiking-Neuron Boundary Study

Date: 2026-07-01

Status: standalone public repository extracted from the anonymous artifact dry
run and then updated with post-boundary architecture probes. It excludes
datasets, checkpoints, raw launcher logs, local build sidecars, private paths,
and unrelated repository material.

This repository is intentionally separated from the SDR-LUT/QK-LUTFormer work.
It contains only the code, run specifications, compact result summaries,
figures, and anonymous paper artifacts needed for the standalone LUT
spiking-neuron boundary study.

## Scope

This artifact supports the paper's bounded diagnostic claims:

- the initial dense trainable LUT-IF route is a registered negative result;
- CNL-LUT-LIF improves only the CIFAR-100 T=4 row in the fixed 6-bit main
  matrix and does not support broad method superiority;
- the CIFAR-100 T=1 failure is precision/state-range sensitive in the fixed
  8-bit diagnostic.
- the RL selector and dense trainable CNL-LUT-LIF architecture probes are
  registered negative results; the structured residual CNL-LUT-LIF probe is
  preregistered but not yet a result in this repository.

The artifact does not claim hardware speed, energy, area, SRAM reduction,
cross-architecture generality, seed stability beyond seed 42, or bit-width
robustness beyond the single registered 8-bit diagnostic.

## Contents

- `lut_if/`: trainable dense LUT-IF and quantized arithmetic LIF modules.
- `tools/`: experiment entry points and direct QKFormer utility dependencies.
- `qkformer_lut/`: lookup utility modules required by the experiment tools.
- `cifar10/`, `cifar100/`: QKFormer model definitions used by the tools.
- `run_specs/`: sanitized preregistered run specs with placeholder paths.
- `results/`: small sanitized reports, CSV summaries, training log, and compact
  metrics files used by the manuscript.
- `figures/`: source and rendered evidence-flow figure.
- `paper/`: anonymous manuscript PDF and reproducibility checklist draft.
- `scripts/`: command templates using environment variables only.
- `checks/SHA256SUMS`: file hashes generated during local dry-run packaging.
- `ACCESS_STATEMENT_DRAFT.md`: draft anonymous-review access boundary; not a
  final public license.

## Not Included

- Datasets. CIFAR-10 and CIFAR-100 should be obtained from the public
  torchvision dataset source.
- Model checkpoints. Exact reproduction requires the retained QKFormer seed-42
  checkpoints referenced by placeholder names in `run_specs/`.
- Raw launcher logs. They are excluded because such logs often contain local
  paths, machine names, or job-control details.
- Any private repository URL, server hostname, IP address, username, token, or
  credential.

## Environment Variables

Set these before running a reproduction command:

```bash
export REPO_ROOT=/path/to/anonymous/qkformer-lut-root
export DATA_DIR=/path/to/cifar-data-root
export CHECKPOINT_DIR=/path/to/qkformer-checkpoints
export RESULT_DIR=/path/to/output-results
export PYTHON=python
export CUDA_VISIBLE_DEVICES=0
```

The artifact should be placed at the root of an anonymous QKFormer-compatible
source tree or merged into that root before running commands. The scripts assume
`PYTHONPATH=$REPO_ROOT`.

## Reproduction Templates

Smoke-first CNL-LUT-LIF diagnostic:

```bash
bash scripts/run_cnl_lut_lif_e0_template.sh smoke_c100_t1_8bit
```

Full CIFAR-100 T=1 8-bit diagnostic:

```bash
bash scripts/run_cnl_lut_lif_e0_template.sh full_c100_t1_8bit
```

Initial LUT-IF proof of concept:

```bash
bash scripts/run_lut_if_poc_template.sh full_c100_t4
```

These templates do not download checkpoints and do not search over seeds,
bit-widths, learning rates, checkpoints, or validation selections.

Structured residual CNL-LUT-LIF architecture probe:

```bash
RUN_MODE=smoke bash scripts/server/run_structured_residual_cnl_lut_lif_probe_template.sh
```

## Notes on Result Labels

The original tool used legacy method labels containing `6bit` in the
CIFAR-100 T=1 boundary `summary.csv`. The registered protocol and compact
metrics in this artifact record `state_bits=8` and `input_bits=8`; the
boundary report explains the label mismatch.

## Release Policy

This repository follows the no-checkpoint review option: provide code, run
specs, small results, and command templates, but do not include model weights.
The included access statement is a draft for anonymous review, not a final
public license.
