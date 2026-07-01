# Anonymous Review Access Statement Draft

Status: draft for local dry-run packaging. This statement is not a final public
license and does not authorize upload, publication, or submission by itself.

## Review Access

This artifact is prepared for anonymous peer review of the accompanying LUT
spiking-neuron boundary-study manuscript. Reviewers may inspect the included
source files, run specifications, command templates, figures, compact metrics,
and result summaries for the purpose of evaluating reproducibility and claim
support during review.

## Included Materials

The dry-run artifact includes:

- method source files for LUT-IF and CNL-LUT-LIF diagnostics;
- direct QKFormer utility files needed by the included experiment entry points;
- sanitized run specifications and command templates;
- compact result summaries and figure source;
- the anonymous manuscript PDF and reproducibility checklist draft.

## Excluded Materials

The dry-run artifact does not include:

- model checkpoints or weights;
- datasets or dataset caches;
- raw launcher logs;
- machine-specific build sidecars;
- credentials, private paths, private repository links, hostnames, or IP
  addresses.

Exact reproduction of the reported full-validation rows requires the retained
QKFormer seed-42 checkpoints named by placeholders in the sanitized run specs.
Those checkpoints require a separate release decision and are not part of this
no-checkpoint review package.

## Third-Party Components

The included code depends on third-party software such as PyTorch, torchvision,
timm, and SpikingJelly. Those projects remain governed by their own licenses.
The QKFormer-compatible model definitions and inherited utility code may also
require an upstream-license review before any public release.

## Final Release

Before any public artifact release or conference upload, the authors must choose
and add an explicit license or access statement for:

- source code;
- result summaries and figures;
- model checkpoints, if released;
- inherited third-party or upstream project code.

Until that decision is made, this file should be treated as an anonymous-review
access draft, not a final license.
