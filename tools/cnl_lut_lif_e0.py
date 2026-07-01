#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch

from lut_if.replace import metadata_summary, replace_with_quantized_arithmetic, set_submodule
from tools.lut_if_poc import (
    build_model,
    build_ranges,
    evaluate_pair,
    freeze,
    generic_module_summary,
    loader_config,
    model_config,
)
from tools.qkformer_lut_current_unit_probe import reset_snn
from tools.qkformer_lut_e2_replace import build_loader
from tools.qkformer_native_affine_lut_probe import (
    QuantizedTransitionLUT,
    discover_lif_targets,
    replace_lif_targets,
)


FAILED_DENSE_LUT_IF_E1_C100_T4 = {
    "candidate_top1": 78.42,
    "drop_top1": 2.60,
    "logit_mse_per_sample": 56.616154718017576,
}


class TCMoment:
    def __init__(self, time_steps: int) -> None:
        self.time_steps = int(time_steps)
        self.sum: Optional[torch.Tensor] = None
        self.sumsq: Optional[torch.Tensor] = None
        self.count: Optional[torch.Tensor] = None

    def _tc_flat(self, x: torch.Tensor) -> torch.Tensor:
        value = x.detach().float().cpu()
        if value.ndim < 3:
            raise RuntimeError(f"expected time-major tensor with channel dimension, got {tuple(value.shape)}")
        if value.shape[0] == self.time_steps:
            # [T, B, C, ...]
            permute = [0, 2] + [dim for dim in range(value.ndim) if dim not in (0, 2)]
            value = value.permute(*permute).contiguous()
            return value.view(value.shape[0], value.shape[1], -1)
        if value.shape[0] % self.time_steps == 0:
            # Some QKFormer attention paths flatten [T, B] into one leading
            # axis before calling MultiStepLIF: [T*B, C, N].
            batch = value.shape[0] // self.time_steps
            channel = value.shape[1]
            return value.reshape(self.time_steps, batch, channel, -1).permute(0, 2, 1, 3).reshape(
                self.time_steps, channel, -1
            )
        raise RuntimeError(
            f"cannot infer temporal layout for shape {tuple(value.shape)} with T={self.time_steps}"
        )

    def update(self, x: torch.Tensor) -> None:
        flat = self._tc_flat(x)
        batch_sum = flat.sum(dim=-1)
        batch_sumsq = flat.square().sum(dim=-1)
        batch_count = torch.full_like(batch_sum, flat.shape[-1])
        if self.sum is None:
            self.sum = torch.zeros_like(batch_sum)
            self.sumsq = torch.zeros_like(batch_sumsq)
            self.count = torch.zeros_like(batch_count)
        if self.sum.shape != batch_sum.shape:
            raise RuntimeError(f"moment shape changed from {tuple(self.sum.shape)} to {tuple(batch_sum.shape)}")
        self.sum += batch_sum
        self.sumsq += batch_sumsq
        self.count += batch_count

    def mean_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.sum is None or self.sumsq is None or self.count is None:
            raise RuntimeError("empty temporal-channel moment")
        count = self.count.clamp_min(1.0)
        mean = self.sum / count
        var = (self.sumsq / count - mean.square()).clamp_min(0.0)
        return mean, var.sqrt().clamp_min(1e-6)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def decoded_uniform(value: torch.Tensor, value_range: Tuple[float, float], levels: int) -> torch.Tensor:
    lo, hi = value_range
    fp32 = value.detach().float()
    if hi <= lo:
        return torch.full_like(fp32, lo)
    scaled = (fp32.clamp(lo, hi) - lo) * ((levels - 1) / (hi - lo))
    index = scaled.round().clamp(0, levels - 1)
    return lo + index * ((hi - lo) / (levels - 1))


def collect_current_moments(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    targets: Mapping[str, torch.nn.Module],
    ranges: Mapping[str, Mapping[str, Tuple[float, float]]],
    *,
    input_bits: int,
    batches: int,
) -> Tuple[int, Dict[str, Dict[str, torch.Tensor]]]:
    args_time_steps = int(getattr(model, "_cnl_time_steps", 4))
    teacher: Dict[str, TCMoment] = defaultdict(lambda: TCMoment(args_time_steps))
    lookup: Dict[str, TCMoment] = defaultdict(lambda: TCMoment(args_time_steps))
    handles = []
    levels = 1 << int(input_bits)

    for name, module in targets.items():
        def make_hook(target_name: str):
            def hook(_module, inputs):
                if not inputs:
                    return
                current = inputs[0]
                teacher[target_name].update(current)
                lookup[target_name].update(decoded_uniform(current, ranges[target_name]["x"], levels))
            return hook
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    seen = 0
    try:
        for images, _target in loader:
            if seen >= batches:
                break
            reset_snn(model)
            model(images.to(device, non_blocking=True))
            seen += 1
    finally:
        for handle in handles:
            handle.remove()
        reset_snn(model)

    moments: Dict[str, Dict[str, torch.Tensor]] = {}
    for name in targets:
        ref_mean, ref_std = teacher[name].mean_std()
        lut_mean, lut_std = lookup[name].mean_std()
        moments[name] = {
            "ref_mean": ref_mean,
            "ref_std": ref_std,
            "lut_mean": lut_mean,
            "lut_std": lut_std,
        }
    return seen, moments


class CurrentNormalizedLUTLIF(QuantizedTransitionLUT):
    def __init__(
        self,
        *,
        moments: Mapping[str, torch.Tensor],
        normalize_current: bool,
        time_steps: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.normalize_current = bool(normalize_current)
        self.time_steps = int(time_steps)
        for key in ("ref_mean", "ref_std", "lut_mean", "lut_std"):
            self.register_buffer(key, moments[key].detach().float().clone())
        self.current_sse_before = 0.0
        self.current_sse_after = 0.0
        self.current_values = 0
        self.norm_clip_count = 0
        self.norm_input_count = 0

    def reset_metrics(self) -> None:
        self.executions = 0
        self.clip_count = 0
        self.input_count = 0
        self.quant_sse = 0.0
        self.current_sse_before = 0.0
        self.current_sse_after = 0.0
        self.current_values = 0
        self.norm_clip_count = 0
        self.norm_input_count = 0

    def _quantize_decode_current(self, value: torch.Tensor) -> torch.Tensor:
        fp32 = value.detach().float()
        lo, hi = self.x_range
        if hi <= lo:
            decoded = torch.full_like(fp32, lo)
            clipped = fp32 != lo
        else:
            clipped = (fp32 < lo) | (fp32 > hi)
            scaled = (fp32.clamp(lo, hi) - lo) * ((self.x_levels - 1) / (hi - lo))
            index = scaled.round().clamp(0, self.x_levels - 1)
            decoded = lo + index * ((hi - lo) / (self.x_levels - 1))
        diff = decoded - fp32
        self.clip_count += int(clipped.sum().item())
        self.input_count += int(fp32.numel())
        self.quant_sse += float(diff.square().sum().item())
        return decoded.to(device=value.device, dtype=value.dtype)

    def _index_current(self, value: torch.Tensor) -> torch.Tensor:
        fp32 = value.detach().float()
        lo, hi = self.x_range
        if hi <= lo:
            clipped = fp32 != lo
            index = torch.zeros_like(fp32, dtype=torch.long)
        else:
            clipped = (fp32 < lo) | (fp32 > hi)
            scaled = (fp32.clamp(lo, hi) - lo) * ((self.x_levels - 1) / (hi - lo))
            index = scaled.round().long().clamp(0, self.x_levels - 1)
        self.norm_clip_count += int(clipped.sum().item())
        self.norm_input_count += int(fp32.numel())
        return index

    def _normalize_step(self, step: int, decoded: torch.Tensor, flattened_tb: bool, batch_size: int) -> torch.Tensor:
        if not self.normalize_current:
            return decoded
        if flattened_tb:
            t = min(int(step) // max(int(batch_size), 1), self.ref_mean.shape[0] - 1)
            channel_count = decoded.shape[0]
        else:
            t = min(int(step), self.ref_mean.shape[0] - 1)
            channel_count = decoded.shape[1] if decoded.ndim >= 2 else self.ref_mean.shape[1]
        if channel_count != self.ref_mean.shape[1]:
            raise RuntimeError(
                f"channel mismatch in CNL-LUT-LIF: current has {channel_count}, moments have {self.ref_mean.shape[1]}"
            )
        if flattened_tb:
            view_shape = [channel_count] + [1] * max(decoded.ndim - 1, 0)
        else:
            view_shape = [1, channel_count] + [1] * max(decoded.ndim - 2, 0)
        ref_mean = self.ref_mean[t].to(device=decoded.device, dtype=decoded.dtype).view(*view_shape)
        ref_std = self.ref_std[t].to(device=decoded.device, dtype=decoded.dtype).view(*view_shape)
        lut_mean = self.lut_mean[t].to(device=decoded.device, dtype=decoded.dtype).view(*view_shape)
        lut_std = self.lut_std[t].to(device=decoded.device, dtype=decoded.dtype).view(*view_shape)
        return (decoded - lut_mean) / lut_std.clamp_min(1e-6) * ref_std + ref_mean

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() <= 1:
            raise ValueError("CurrentNormalizedLUTLIF expects [T, ...] input")
        self.executions += 1
        if isinstance(self.v, float):
            v = torch.zeros_like(x_seq[0])
            if self.v != 0.0:
                v.fill_(float(self.v))
        else:
            v = self.v.to(device=x_seq.device, dtype=x_seq.dtype)
        spike_table, state_table = self._tables(x_seq.device, x_seq.dtype)
        flattened_tb = bool(x_seq.shape[0] != self.time_steps and x_seq.shape[0] % self.time_steps == 0)
        batch_size = int(x_seq.shape[0] // self.time_steps) if flattened_tb else int(x_seq.shape[1])
        spikes = []
        states = []
        for step in range(x_seq.shape[0]):
            original = x_seq[step]
            decoded = self._quantize_decode_current(original)
            normalized = self._normalize_step(step, decoded, flattened_tb, batch_size)
            self.current_sse_before += float((decoded.float() - original.detach().float()).square().sum().item())
            self.current_sse_after += float((normalized.detach().float() - original.detach().float()).square().sum().item())
            self.current_values += int(original.numel())
            x_index = self._index_current(normalized)
            v_index = self._quantize(v, self.v_range, self.v_levels)
            flat_index = v_index * self.x_levels + x_index
            spike = spike_table[flat_index]
            v = state_table[flat_index]
            spikes.append(spike.unsqueeze(0))
            states.append(v.unsqueeze(0))
        self.v = states[-1].squeeze(0).detach().clone()
        self.v_seq = torch.cat(states, dim=0)
        return torch.cat(spikes, dim=0)

    def summary(self) -> Dict[str, float]:
        base = super().summary()
        values = max(self.current_values, 1)
        base.update(
            {
                "normalize_current": float(self.normalize_current),
                "current_mse_before_norm": self.current_sse_before / values,
                "current_mse_after_norm": self.current_sse_after / values,
                "norm_clip_rate": self.norm_clip_count / self.norm_input_count if self.norm_input_count else 0.0,
                "metadata_bytes": float(self.table_entries() * 4 + 4 * self.ref_mean.numel() * 4),
                "table_entries": float(self.table_entries()),
            }
        )
        return base


def replace_with_cnl_lut_lif(
    model: torch.nn.Module,
    targets: Mapping[str, torch.nn.Module],
    ranges: Mapping[str, Mapping[str, Tuple[float, float]]],
    moments: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    state_bits: int,
    input_bits: int,
    normalize_current: bool,
    time_steps: int,
) -> Dict[str, CurrentNormalizedLUTLIF]:
    replacements: Dict[str, CurrentNormalizedLUTLIF] = {}
    for name, original in list(targets.items()):
        common = {
            "tau": float(getattr(original, "tau")),
            "decay_input": bool(getattr(original, "decay_input")),
            "v_threshold": float(getattr(original, "v_threshold")),
            "v_reset": getattr(original, "v_reset"),
        }
        replacement = CurrentNormalizedLUTLIF(
            **common,
            x_range=ranges[name]["x"],
            v_range=ranges[name]["v"],
            bits=state_bits,
            moments=moments[name],
            normalize_current=normalize_current,
            time_steps=time_steps,
        )
        replacement.to(next(model.parameters()).device)
        set_submodule(model, name, replacement)
        replacements[name] = replacement
    return replacements


def save_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_current_diagnostics(modules: Mapping[str, CurrentNormalizedLUTLIF]) -> Dict[str, float]:
    before = sum(module.current_sse_before for module in modules.values())
    after = sum(module.current_sse_after for module in modules.values())
    values = sum(module.current_values for module in modules.values())
    norm_clip = sum(module.norm_clip_count for module in modules.values())
    norm_values = sum(module.norm_input_count for module in modules.values())
    return {
        "current_mse_before_normalization": before / values if values else 0.0,
        "current_mse_after_normalization": after / values if values else 0.0,
        "normalization_clip_rate": norm_clip / norm_values if norm_values else 0.0,
    }


def build_report(metrics: Mapping[str, object]) -> str:
    gate = metrics["gate"]
    rows = metrics["summary_rows"]
    table = "\n".join(
        f"| {row['method']} | {row['candidate_top1']:.4f} | {row['drop_top1']:.4f} | {row['logit_mse_per_sample']:.6f} |"
        for row in rows
    )
    return f"""# CNL-LUT-LIF E0

Status: **{gate['verdict']}**

## Scope

- QKFormer {metrics['protocol']['family']}, seed {metrics['protocol']['seed']}, T={metrics['protocol']['time_step']}
- All discovered MultiStepLIF targets, fixed {metrics['protocol']['state_bits']}-bit state/input transition LUT
- Current-normalization ablation only; no bit-width, seed, checkpoint, stage, or learning-rate search

## Results

| Method | Acc@1 | Drop | Logit MSE/sample |
|---|---:|---:|---:|
{table}

## Gate

- Beats post-hoc dense LUT on Acc@1 drop or logit MSE: **{gate['beats_posthoc_on_top1_or_logit_mse']}**
- Drop below the same-run post-hoc dense LUT: **{gate['drop_below_same_run_posthoc']}**
- Drop below the external registered threshold: **{gate['drop_below_external_threshold']}**
- Current MSE improves over no-normalization: **{gate['current_mse_improves_over_no_norm']}**
- All requested targets executed: **{gate['all_requested_targets_executed']}**

## Claim Boundary

This run can only test whether current-normalized lookup current improves the
registered all-LIF replacement tradeoff on this retained QKFormer setting. It
does not establish hardware efficiency, compactness beyond metadata accounting,
or broad architecture generalization.
"""


def run(args) -> Dict[str, object]:
    set_seed(args.seed)
    root = Path(args.root).resolve()
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.local_cuda_index}" if torch.cuda.is_available() else "cpu")

    train_loader = build_loader(loader_config(args, "train", True), device)
    validation_loader = build_loader(loader_config(args, "validation", False), device)

    clean, clean_checkpoint = build_model(root, args, device)
    freeze(clean)
    clean.eval()
    setattr(clean, "_cnl_time_steps", int(args.time_step))
    calibration_batches, ranges, range_adjustments = build_ranges(clean, train_loader, device, args)
    clean_targets = discover_lif_targets(clean)
    moment_batches, moments = collect_current_moments(
        clean,
        train_loader,
        device,
        clean_targets,
        ranges,
        input_bits=args.input_bits,
        batches=args.calib_batches,
    )

    posthoc_model, posthoc_checkpoint = build_model(root, args, device)
    quant_model, quant_checkpoint = build_model(root, args, device)
    no_norm_model, no_norm_checkpoint = build_model(root, args, device)
    cnl_model, cnl_checkpoint = build_model(root, args, device)
    for model in (posthoc_model, quant_model, no_norm_model, cnl_model):
        freeze(model)

    posthoc_replacements = replace_lif_targets(
        posthoc_model,
        discover_lif_targets(posthoc_model),
        lif_lut_bits=args.state_bits,
        lif_ranges=ranges,
    )
    quant_replacements = replace_with_quantized_arithmetic(
        quant_model,
        discover_lif_targets(quant_model),
        ranges,
        state_bits=args.state_bits,
        input_bits=args.input_bits,
    )
    no_norm_replacements = replace_with_cnl_lut_lif(
        no_norm_model,
        discover_lif_targets(no_norm_model),
        ranges,
        moments,
        state_bits=args.state_bits,
        input_bits=args.input_bits,
        normalize_current=False,
        time_steps=args.time_step,
    )
    cnl_replacements = replace_with_cnl_lut_lif(
        cnl_model,
        discover_lif_targets(cnl_model),
        ranges,
        moments,
        state_bits=args.state_bits,
        input_bits=args.input_bits,
        normalize_current=True,
        time_steps=args.time_step,
    )

    posthoc = evaluate_pair(clean, posthoc_model, validation_loader, device, posthoc_replacements, args.max_eval_batches)
    quant = evaluate_pair(clean, quant_model, validation_loader, device, quant_replacements, args.max_eval_batches)
    no_norm = evaluate_pair(clean, no_norm_model, validation_loader, device, no_norm_replacements, args.max_eval_batches)
    cnl = evaluate_pair(clean, cnl_model, validation_loader, device, cnl_replacements, args.max_eval_batches)

    posthoc_storage = generic_module_summary(posthoc_replacements)
    quant_storage = metadata_summary(quant_replacements)
    no_norm_storage = metadata_summary(no_norm_replacements)
    cnl_storage = metadata_summary(cnl_replacements)
    expected_targets = len(cnl_replacements)
    all_targets_executed = bool(
        posthoc_storage["modules_executed"] == expected_targets
        and quant_storage["modules_executed"] == expected_targets
        and no_norm_storage["modules_executed"] == expected_targets
        and cnl_storage["modules_executed"] == expected_targets
    )
    no_norm_diag = aggregate_current_diagnostics(no_norm_replacements)
    cnl_diag = aggregate_current_diagnostics(cnl_replacements)
    beats_posthoc = (
        cnl["candidate_top1"] > posthoc["candidate_top1"]
        or cnl["logit_mse_per_sample"] < posthoc["logit_mse_per_sample"]
    )
    drop_below_same_run_posthoc = cnl["drop_top1"] < posthoc["drop_top1"]
    drop_below_external_threshold = cnl["drop_top1"] < args.drop_threshold
    current_mse_improves = (
        cnl_diag["current_mse_after_normalization"]
        < no_norm_diag["current_mse_after_normalization"]
    )
    smoke_mode = args.max_eval_batches is not None
    smoke_pass = bool(all_targets_executed)
    gate_pass = bool(
        beats_posthoc
        and drop_below_same_run_posthoc
        and drop_below_external_threshold
        and current_mse_improves
        and all_targets_executed
    )

    summary_rows = [
        {"method": "posthoc_dense_transition_LUT_6bit", **posthoc},
        {"method": "quantized_arithmetic_LIF_6bit", **quant},
        {"method": "CNL_LUT_LIF_without_normalization", **no_norm},
        {"method": "CNL_LUT_LIF_with_normalization", **cnl},
    ]
    if args.include_failed_dense_e1:
        summary_rows.insert(2, {"method": "dense_trainable_LUT_IF_E1_failed", **FAILED_DENSE_LUT_IF_E1_C100_T4})
    metrics = {
        "experiment": f"cnl_lut_lif_main_{args.family}_t{args.time_step}_seed{args.seed}",
        "status": "completed",
        "protocol": {
            "family": args.family,
            "num_classes": args.num_classes,
            "seed": args.seed,
            "time_step": args.time_step,
            "state_bits": args.state_bits,
            "input_bits": args.input_bits,
            "calibration_batches": calibration_batches,
            "moment_batches": moment_batches,
            "max_eval_batches": args.max_eval_batches,
            "range_adjustments": range_adjustments,
            "checkpoint": args.checkpoint,
            "data_dir": args.data_dir,
        },
        "checkpoint_load": {
            "clean": clean_checkpoint,
            "posthoc": posthoc_checkpoint,
            "quantized_arithmetic": quant_checkpoint,
            "no_norm": no_norm_checkpoint,
            "cnl": cnl_checkpoint,
        },
        "results": {
            "posthoc_dense_transition_LUT_6bit": posthoc,
            "quantized_arithmetic_LIF_6bit": quant,
            "CNL_LUT_LIF_without_normalization": no_norm,
            "CNL_LUT_LIF_with_normalization": cnl,
        },
        "diagnostics": {
            "no_normalization": no_norm_diag,
            "with_normalization": cnl_diag,
        },
        "storage": {
            "posthoc": posthoc_storage,
            "quantized_arithmetic": quant_storage,
            "no_normalization": no_norm_storage,
            "with_normalization": cnl_storage,
        },
        "summary_rows": summary_rows,
        "gate": {
            "beats_posthoc_on_top1_or_logit_mse": beats_posthoc,
            "drop_below_same_run_posthoc": drop_below_same_run_posthoc,
            "drop_below_external_threshold": drop_below_external_threshold,
            "current_mse_improves_over_no_norm": current_mse_improves,
            "all_requested_targets_executed": all_targets_executed,
            "expected_targets": expected_targets,
            "drop_threshold": args.drop_threshold,
            "smoke_mode": smoke_mode,
            "pass": smoke_pass if smoke_mode else gate_pass,
            "formal_full_gate_pass": gate_pass,
            "verdict": (
                ("SMOKE-PASS" if smoke_pass else "SMOKE-FAIL")
                if smoke_mode
                else ("GO" if gate_pass else "NO-GO")
            ),
        },
    }
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, default=str)
    save_csv(result_dir / "summary.csv", summary_rows)
    (result_dir / "report.md").write_text(build_report(metrics), encoding="utf-8")
    return metrics


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CNL-LUT-LIF E0 registered follow-up")
    p.add_argument("--root", default=".")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--result-dir", required=True)
    p.add_argument("--family", choices=["cifar10", "cifar100"], default="cifar100")
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--local-cuda-index", type=int, default=0)
    p.add_argument("--time-step", type=int, default=4)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--calib-batches", type=int, default=128)
    p.add_argument("--range-margin", type=float, default=0.05)
    p.add_argument("--state-bits", type=int, default=6)
    p.add_argument("--input-bits", type=int, default=6)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--drop-threshold", type=float, default=2.06)
    p.add_argument("--include-failed-dense-e1", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
