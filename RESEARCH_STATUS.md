# Research Status

Status: `CONDITIONAL_PASS_LOCAL_PACKAGE`.

The current story is coherent if framed as a boundary study rather than a broad
positive method paper:

1. Direct dense LUT-IF replacement is a registered negative result.
2. CNL-LUT-LIF helps in the CIFAR-100 `T=4` all-LIF replacement row under the
   fixed 6-bit protocol.
3. The severe CIFAR-100 `T=1` 6-bit failure is explained by a registered 8-bit
   precision/state-range diagnostic.
4. The paper's contribution is the controlled diagnostic protocol and evidence
   boundary, not a claim that LUT-LIF is generally superior.

## Main Evidence

- Dense trainable LUT-IF proof of concept on QKFormer CIFAR-100 `T=4`, seed 42:
  teacher `81.02`, trained Dense LUT-IF `78.42`, post-hoc dense LUT `78.96`,
  quantized arithmetic LIF `78.90`. Gate: `NO-GO`.
- CNL-LUT-LIF main matrix, fixed seed 42 and 6-bit state/input:
  - CIFAR-100 `T=1`: teacher `77.58`, normalized CNL-LUT-LIF `70.79`, drop
    `6.79`. Gate: fail.
  - CIFAR-100 `T=4`: teacher `81.02`, post-hoc dense LUT `79.55`, quantized
    arithmetic LIF `79.61`, normalized CNL-LUT-LIF `79.77`, drop `1.25`.
    Gate: per-run `GO`.
  - CIFAR-10 `T=1`: teacher `94.90`, normalized CNL-LUT-LIF `94.53`, tied
    with post-hoc.
  - CIFAR-10 `T=4`: teacher `95.73`, normalized CNL-LUT-LIF `95.64`, tied
    with post-hoc.
  Matrix verdict: `NO-GO_FOR_BROAD_MAIN_MATRIX`.
- CIFAR-100 `T=1` 8-bit boundary diagnostic:
  teacher `77.58`, 8-bit post-hoc dense LUT `77.12`, 8-bit quantized
  arithmetic LIF `77.07`, 8-bit normalized CNL-LUT-LIF `77.12`. Primary
  diagnostic gate: `GO`; secondary CNL superiority gate: `NO-GO`.

## Current Readiness

The local anonymous paper and artifact packages have passed local verification,
including source, structured-file, privacy, PDF, archive, syntax, and package
consistency checks in the source project.

Remaining blockers before any public release or submission:

- explicit user approval for checkpoint policy;
- explicit user approval for license/access statement;
- explicit upload/submission approval;
- final live portal and official-policy freshness check.

No remote-server experiment is planned or run in this extraction step.
