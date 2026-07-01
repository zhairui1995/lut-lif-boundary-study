# Artifact Safety Report

Date: 2026-06-29

Package: `anonymous_artifact_dry_run_20260629`

Status: conditional local dry-run pass. This package has not been uploaded,
published, or submitted.

## Inclusion Boundary

Included:

- LUT-IF and CNL-LUT-LIF source files needed for the boundary study.
- Direct QKFormer utility dependencies used by the included entry points.
- CIFAR-10/CIFAR-100 model definitions.
- Sanitized run specs with placeholder environment variables.
- Small result summaries, compact metrics, and figure source/rendered figure.
- Anonymous manuscript PDF and reproducibility checklist draft.
- Anonymous-review access statement draft.

Excluded:

- datasets;
- checkpoints and model weights;
- raw launcher logs;
- build sidecars such as `.aux`, `.log`, `.fls`, and `.fdb_latexmk`;
- `.git`, `.ai-bridge`, editor metadata, and unrelated paper materials;
- private paths, usernames, hostnames, IP addresses, private repository owner
  strings, API keys, tokens, cookies, or private keys.

## Checks Run

- Source syntax: `python3 -m py_compile` on included Python files.
- Structured files: PyYAML parse for all run specs and JSON parse for compact
  metrics.
- Size scan: no file larger than 10 MiB.
- Build-sidecar scan: no `.aux`, `.log`, `.fls`, `.fdb_latexmk`, or
  `launcher.log` files.
- Text privacy scan: no absolute home-directory paths, local usernames, server
  IPs, or private repository owner strings in text files.
- PDF privacy scan: Python PDF text extraction found no absolute home-directory
  paths, local usernames, server IPs, or private repository owner strings in
  the included manuscript and checklist PDFs.

## Residual Conditions

- Exact result reproduction still requires retained QKFormer checkpoints that
  are not included in this dry run.
- No final public release license is included. The access statement is a draft
  for anonymous review only.
- Final upload or submission requires explicit user approval.
