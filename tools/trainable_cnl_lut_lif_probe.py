#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lut_if.neuron import DenseLUTIFNeuron
from lut_if.replace import (
    metadata_summary,
    replace_with_quantized_arithmetic,
    reset_metrics,
    set_submodule,
    trainable_parameters,
)
from tools.cnl_lut_lif_e0 import collect_current_moments, replace_with_cnl_lut_lif
from tools.lut_if_poc import (
    build_model,
    build_ranges,
    enable_lutif_training,
    evaluate_pair,
    freeze,
    generic_module_summary,
    loader_config,
    top1,
)
from tools.qkformer_lut_current_unit_probe import reset_snn
from tools.qkformer_lut_e2_replace import build_loader
from tools.qkformer_native_affine_lut_probe import discover_lif_targets, replace_lif_targets


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class TrainableCNLDenseLUTIF(DenseLUTIFNeuron):
    """Dense trainable LUT-IF with CNL current normalization before lookup."""

    def __init__(
        self,
        *,
        moments: Mapping[str, torch.Tensor],
        time_steps: int,
        learn_norm_affine: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.time_steps = int(time_steps)
        self.learn_norm_affine = bool(learn_norm_affine)
        for key in ("ref_mean", "ref_std", "lut_mean", "lut_std"):
            self.register_buffer(key, moments[key].detach().float().clone())
        if self.learn_norm_affine:
            self.norm_scale = torch.nn.Parameter(torch.ones_like(self.ref_mean))
            self.norm_bias = torch.nn.Parameter(torch.zeros_like(self.ref_mean))
        else:
            self.register_buffer("norm_scale", torch.ones_like(self.ref_mean))
            self.register_buffer("norm_bias", torch.zeros_like(self.ref_mean))
        self.current_sse_before = 0.0
        self.current_sse_after = 0.0
        self.current_values = 0

    def reset_metrics(self) -> None:
        super().reset_metrics()
        self.current_sse_before = 0.0
        self.current_sse_after = 0.0
        self.current_values = 0

    def _normalize_step(self, step: int, current: torch.Tensor, flattened_tb: bool, batch_size: int) -> torch.Tensor:
        if flattened_tb:
            t = min(int(step) // max(int(batch_size), 1), self.ref_mean.shape[0] - 1)
            channel_count = current.shape[0]
            view_shape = [channel_count] + [1] * max(current.ndim - 1, 0)
        else:
            t = min(int(step), self.ref_mean.shape[0] - 1)
            channel_count = current.shape[1] if current.ndim >= 2 else self.ref_mean.shape[1]
            view_shape = [1, channel_count] + [1] * max(current.ndim - 2, 0)
        if channel_count != self.ref_mean.shape[1]:
            raise RuntimeError(
                f"channel mismatch in TrainableCNLDenseLUTIF: current has {channel_count}, moments have {self.ref_mean.shape[1]}"
            )
        ref_mean = self.ref_mean[t].to(device=current.device, dtype=current.dtype).view(*view_shape)
        ref_std = self.ref_std[t].to(device=current.device, dtype=current.dtype).view(*view_shape)
        lut_mean = self.lut_mean[t].to(device=current.device, dtype=current.dtype).view(*view_shape)
        lut_std = self.lut_std[t].to(device=current.device, dtype=current.dtype).view(*view_shape)
        scale = self.norm_scale[t].to(device=current.device, dtype=current.dtype).view(*view_shape)
        bias = self.norm_bias[t].to(device=current.device, dtype=current.dtype).view(*view_shape)
        normalized = (current - lut_mean) / lut_std.clamp_min(1e-6) * ref_std + ref_mean
        return normalized * scale + bias

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim < 2:
            raise ValueError("TrainableCNLDenseLUTIF expects a time-major tensor")
        self.executions += 1
        flattened_tb = bool(x_seq.shape[0] != self.time_steps and x_seq.shape[0] % self.time_steps == 0)
        batch_size = int(x_seq.shape[0] // self.time_steps) if flattened_tb else int(x_seq.shape[1])
        spikes = []
        states = []
        for step in range(x_seq.shape[0]):
            original = x_seq[step]
            normalized = self._normalize_step(step, original, flattened_tb, batch_size)
            if not self.training:
                self.current_sse_before += float((original.detach().float() - normalized.detach().float()).square().sum().item())
                self.current_sse_after += float((normalized.detach().float() - original.detach().float()).square().sum().item())
                self.current_values += int(original.numel())
            spikes.append(self.single_step_forward(normalized))
            states.append(self._state_tensor(normalized))
        self.v_seq = torch.stack(states, dim=0)
        return torch.stack(spikes, dim=0)

    def regularization(self) -> Dict[str, torch.Tensor]:
        base = super().regularization()
        if self.learn_norm_affine:
            base["norm_affine"] = (self.norm_scale - 1.0).square().mean() + self.norm_bias.square().mean()
        else:
            base["norm_affine"] = self.transition_table.new_tensor(0.0)
        return base

    def metadata_bytes(self) -> int:
        base = super().metadata_bytes()
        moment_bytes = 4 * self.ref_mean.numel() * 4
        affine_bytes = 2 * self.ref_mean.numel() * 4 if self.learn_norm_affine else 0
        return int(base + moment_bytes + affine_bytes)

    def summary(self) -> Dict[str, float]:
        base = super().summary()
        values = max(self.current_values, 1)
        base.update(
            {
                "learn_norm_affine": float(self.learn_norm_affine),
                "current_mse_after_cnl": self.current_sse_after / values,
                "metadata_bytes": float(self.metadata_bytes()),
                "norm_scale_rms_delta": float((self.norm_scale.detach() - 1.0).square().mean().sqrt().item()),
                "norm_bias_rms": float(self.norm_bias.detach().square().mean().sqrt().item()),
            }
        )
        return base


def replace_with_trainable_cnl_lutif(
    model: torch.nn.Module,
    targets: Mapping[str, torch.nn.Module],
    ranges: Mapping[str, Mapping[str, Tuple[float, float]]],
    moments: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    state_bits: int,
    input_bits: int,
    time_steps: int,
    surrogate_slope: float,
    learn_norm_affine: bool,
) -> Dict[str, TrainableCNLDenseLUTIF]:
    replacements: Dict[str, TrainableCNLDenseLUTIF] = {}
    for name, original in list(targets.items()):
        replacement = TrainableCNLDenseLUTIF(
            tau=float(getattr(original, "tau")),
            decay_input=bool(getattr(original, "decay_input")),
            v_threshold=float(getattr(original, "v_threshold")),
            v_reset=getattr(original, "v_reset"),
            x_range=ranges[name]["x"],
            v_range=ranges[name]["v"],
            state_bits=state_bits,
            input_bits=input_bits,
            surrogate_slope=surrogate_slope,
            learn_threshold=True,
            moments=moments[name],
            time_steps=time_steps,
            learn_norm_affine=learn_norm_affine,
        )
        replacement.to(next(model.parameters()).device)
        set_submodule(model, name, replacement)
        replacements[name] = replacement
    return replacements


def regularization_terms(modules: Mapping[str, TrainableCNLDenseLUTIF]) -> Dict[str, torch.Tensor]:
    terms = [module.regularization() for module in modules.values()]
    if not terms:
        raise ValueError("no Trainable CNL-LUT-IF modules were provided")
    return {key: torch.stack([term[key] for term in terms]).mean() for key in terms[0]}


def train_one_epoch(teacher, candidate, replacements, loader, optimizer, scaler, device, args, epoch: int):
    teacher.eval()
    enable_lutif_training(candidate, replacements)
    totals = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "delta_l2": 0.0, "smoothness": 0.0, "norm_affine": 0.0}
    samples = 0
    batches = 0
    correct = 0
    for batch_index, (images, target) in enumerate(loader):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        reset_snn(teacher)
        with torch.no_grad():
            teacher_logits = teacher(images)
        reset_snn(candidate)
        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            logits = candidate(images)
            ce = F.cross_entropy(logits.float(), target, label_smoothing=args.label_smoothing)
            temperature = args.distill_temperature
            kd = F.kl_div(
                F.log_softmax(logits.float() / temperature, dim=1),
                F.softmax(teacher_logits.float() / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature * temperature)
            reg = regularization_terms(replacements)
            loss = (
                ce
                + args.distill_weight * kd
                + args.delta_l2_weight * reg["delta_l2"]
                + args.smoothness_weight * reg["neighbor_smoothness"]
                + args.norm_affine_weight * reg["norm_affine"]
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite trainable CNL-LUT-IF loss at epoch={epoch} batch={batch_index}: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(trainable_parameters(replacements)), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        count = int(target.numel())
        samples += count
        batches += 1
        correct += top1(logits.detach(), target)
        totals["loss"] += float(loss.detach().item()) * count
        totals["ce"] += float(ce.detach().item()) * count
        totals["kd"] += float(kd.detach().item()) * count
        totals["delta_l2"] += float(reg["delta_l2"].detach().item()) * count
        totals["smoothness"] += float(reg["neighbor_smoothness"].detach().item()) * count
        totals["norm_affine"] += float(reg["norm_affine"].detach().item()) * count
        reset_snn(candidate)
    if samples == 0:
        raise RuntimeError("training loader produced no samples")
    return {
        "epoch": epoch,
        "batches": batches,
        "samples": samples,
        "train_top1": 100.0 * correct / samples,
        **{key: value / samples for key, value in totals.items()},
    }


def build_report(metrics: Mapping[str, object]) -> str:
    gate = metrics["gate"]
    rows = metrics["summary_rows"]
    table = "\n".join(
        f"| {row['method']} | {row['candidate_top1']:.4f} | {row['drop_top1']:.4f} | {row['logit_mse_per_sample']:.6f} |"
        for row in rows
    )
    return f"""# Trainable CNL-LUT-LIF Probe

Status: **{gate['verdict']}**

## Scope

- QKFormer {metrics['protocol']['family']}, seed {metrics['protocol']['seed']}, T={metrics['protocol']['time_step']}
- All discovered MultiStepLIF targets
- Fixed {metrics['protocol']['state_bits']}-bit state/input dense trainable transition LUT
- Frozen backbone; trainable transition tables, thresholds, and CNL affine parameters

## Results

| Method | Acc@1 | Drop | Logit MSE/sample |
|---|---:|---:|---:|
{table}

## Gate

- Beats fixed CNL on Acc@1 or logit MSE: **{gate['beats_fixed_cnl_on_top1_or_logit_mse']}**
- Drop below fixed CNL: **{gate['drop_below_fixed_cnl']}**
- Drop within threshold: **{gate['drop_within_threshold']}**
- All requested targets executed: **{gate['all_requested_targets_executed']}**

## Claim Boundary

This run tests whether QK-LUT-inspired current normalization can rescue the
trainable LUT-LIF objective. It does not establish broad architecture transfer,
near-SOTA accuracy, or hardware efficiency unless the fixed full-validation
gate passes and is followed by cross-setting evidence.
"""


def run(args):
    set_seed(args.seed)
    root = Path(args.root).resolve()
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.local_cuda_index}" if torch.cuda.is_available() else "cpu")

    train_loader = build_loader(loader_config(args, "train", True), device)
    val_loader = build_loader(loader_config(args, "validation", False), device)

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
    fixed_cnl_model, fixed_cnl_checkpoint = build_model(root, args, device)
    trainable_model, trainable_checkpoint = build_model(root, args, device)
    for model in (posthoc_model, quant_model, fixed_cnl_model, trainable_model):
        freeze(model)

    posthoc_replacements = replace_lif_targets(posthoc_model, discover_lif_targets(posthoc_model), args.state_bits, ranges)
    quant_replacements = replace_with_quantized_arithmetic(
        quant_model, discover_lif_targets(quant_model), ranges, state_bits=args.state_bits, input_bits=args.input_bits
    )
    fixed_cnl_replacements = replace_with_cnl_lut_lif(
        fixed_cnl_model,
        discover_lif_targets(fixed_cnl_model),
        ranges,
        moments,
        state_bits=args.state_bits,
        input_bits=args.input_bits,
        normalize_current=True,
        time_steps=args.time_step,
    )
    trainable_replacements = replace_with_trainable_cnl_lutif(
        trainable_model,
        discover_lif_targets(trainable_model),
        ranges,
        moments,
        state_bits=args.state_bits,
        input_bits=args.input_bits,
        time_steps=args.time_step,
        surrogate_slope=args.surrogate_slope,
        learn_norm_affine=args.learn_norm_affine,
    )

    posthoc = evaluate_pair(clean, posthoc_model, val_loader, device, posthoc_replacements, args.max_eval_batches)
    quant = evaluate_pair(clean, quant_model, val_loader, device, quant_replacements, args.max_eval_batches)
    fixed_cnl = evaluate_pair(clean, fixed_cnl_model, val_loader, device, fixed_cnl_replacements, args.max_eval_batches)
    initial = evaluate_pair(clean, trainable_model, val_loader, device, trainable_replacements, args.max_eval_batches)

    parameters = list(trainable_parameters(trainable_replacements))
    if not parameters:
        raise RuntimeError("trainable CNL-LUT-IF replacement produced no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    training_log = []
    best_eval = None
    best_top1 = -math.inf
    best_state = None
    for epoch in range(1, args.epochs + 1):
        row = train_one_epoch(clean, trainable_model, trainable_replacements, train_loader, optimizer, scaler, device, args, epoch)
        validation = evaluate_pair(clean, trainable_model, val_loader, device, trainable_replacements, args.max_eval_batches)
        row.update({f"val_{key}": value for key, value in validation.items()})
        training_log.append(row)
        if validation["candidate_top1"] > best_top1:
            best_top1 = validation["candidate_top1"]
            best_eval = validation
            best_state = {
                name: {key: value.detach().cpu() for key, value in module.state_dict().items()}
                for name, module in trainable_replacements.items()
            }
    if best_eval is None or best_state is None:
        raise RuntimeError("no best trainable CNL-LUT-IF state was recorded")
    torch.save(
        {"protocol": vars(args), "replacements": best_state, "best_evaluation": best_eval},
        result_dir / "trainable_cnl_lut_lif_best.pt",
    )

    fixed_cnl_storage = metadata_summary(fixed_cnl_replacements)
    trainable_storage = metadata_summary(trainable_replacements)
    posthoc_storage = generic_module_summary(posthoc_replacements)
    quant_storage = metadata_summary(quant_replacements)
    expected_targets = len(trainable_replacements)
    all_targets_executed = bool(
        posthoc_storage["modules_executed"] == expected_targets
        and quant_storage["modules_executed"] == expected_targets
        and fixed_cnl_storage["modules_executed"] == expected_targets
        and trainable_storage["modules_executed"] == expected_targets
    )
    beats_fixed_cnl = (
        best_eval["candidate_top1"] > fixed_cnl["candidate_top1"]
        or best_eval["logit_mse_per_sample"] < fixed_cnl["logit_mse_per_sample"]
    )
    drop_below_fixed_cnl = best_eval["drop_top1"] < fixed_cnl["drop_top1"]
    drop_within = best_eval["drop_top1"] <= args.max_drop
    smoke_mode = args.max_eval_batches is not None
    smoke_pass = bool(all_targets_executed and len(training_log) > 0)
    gate_pass = bool(beats_fixed_cnl and drop_below_fixed_cnl and drop_within and all_targets_executed)

    summary_rows = [
        {"method": "posthoc_dense_transition_LUT", **posthoc},
        {"method": "quantized_arithmetic_LIF", **quant},
        {"method": "fixed_CNL_LUT_LIF", **fixed_cnl},
        {"method": "trainable_CNL_LUT_LIF_initial", **initial},
        {"method": "trainable_CNL_LUT_LIF_final", **best_eval},
    ]
    metrics = {
        "experiment": "trainable_cnl_lut_lif_probe",
        "status": "completed",
        "protocol": {
            "family": args.family,
            "seed": args.seed,
            "time_step": args.time_step,
            "state_bits": args.state_bits,
            "input_bits": args.input_bits,
            "epochs": args.epochs,
            "max_train_batches": args.max_train_batches,
            "max_eval_batches": args.max_eval_batches,
            "calibration_batches": calibration_batches,
            "moment_batches": moment_batches,
            "learn_norm_affine": args.learn_norm_affine,
            "range_adjustments": range_adjustments,
            "checkpoint": args.checkpoint,
            "data_dir": args.data_dir,
        },
        "checkpoint_load": {
            "clean": clean_checkpoint,
            "posthoc": posthoc_checkpoint,
            "quant": quant_checkpoint,
            "fixed_cnl": fixed_cnl_checkpoint,
            "trainable": trainable_checkpoint,
        },
        "summary_rows": summary_rows,
        "training": training_log,
        "storage": {
            "posthoc": posthoc_storage,
            "quantized_arithmetic": quant_storage,
            "fixed_cnl": fixed_cnl_storage,
            "trainable_cnl": trainable_storage,
        },
        "gate": {
            "beats_fixed_cnl_on_top1_or_logit_mse": beats_fixed_cnl,
            "drop_below_fixed_cnl": drop_below_fixed_cnl,
            "drop_within_threshold": drop_within,
            "all_requested_targets_executed": all_targets_executed,
            "expected_targets": expected_targets,
            "max_drop": args.max_drop,
            "smoke_mode": smoke_mode,
            "pass": smoke_pass if smoke_mode else gate_pass,
            "formal_full_gate_pass": gate_pass,
            "verdict": ("SMOKE-PASS" if smoke_pass else "SMOKE-FAIL") if smoke_mode else ("GO" if gate_pass else "NO-GO"),
        },
    }
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, default=str)
    save_csv(result_dir / "summary.csv", summary_rows)
    save_csv(result_dir / "training_log.csv", training_log)
    (result_dir / "report.md").write_text(build_report(metrics), encoding="utf-8")
    print(json.dumps({"result_dir": str(result_dir), "verdict": metrics["gate"]["verdict"]}, sort_keys=True))
    return metrics


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Trainable CNL-LUT-LIF architecture probe")
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
    p.add_argument("--calib-batches", type=int, default=32)
    p.add_argument("--range-margin", type=float, default=0.05)
    p.add_argument("--state-bits", type=int, default=6)
    p.add_argument("--input-bits", type=int, default=6)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-train-batches", type=int, default=128)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--distill-weight", type=float, default=0.5)
    p.add_argument("--distill-temperature", type=float, default=2.0)
    p.add_argument("--delta-l2-weight", type=float, default=1e-4)
    p.add_argument("--smoothness-weight", type=float, default=1e-4)
    p.add_argument("--norm-affine-weight", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--surrogate-slope", type=float, default=2.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-drop", type=float, default=1.25)
    p.add_argument("--learn-norm-affine", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
