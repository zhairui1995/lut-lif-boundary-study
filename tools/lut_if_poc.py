#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from lut_if.replace import (
    metadata_summary,
    regularization_terms,
    replace_with_dense_lutif,
    replace_with_quantized_arithmetic,
    reset_metrics,
    trainable_parameters,
)
from tools.qkformer_lut_current_unit_probe import reset_snn
from tools.qkformer_lut_e2_replace import (
    build_cifar10_model,
    build_loader,
    load_checkpoint_if_available,
)
from tools.qkformer_native_affine_lut_probe import (
    collect_lif_ranges,
    discover_lif_targets,
    expanded_range,
    replace_lif_targets,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_config(args) -> Dict[str, object]:
    family = str(getattr(args, "family", "cifar100"))
    num_classes_value = getattr(args, "num_classes", None)
    num_classes = int(num_classes_value) if num_classes_value is not None else (100 if family == "cifar100" else 10)
    return {
        "family": family,
        "img_size": 32,
        "patch_size": 4,
        "dim": args.dim,
        "num_heads": 8,
        "mlp_ratio": 4,
        "in_channels": 3,
        "num_classes": num_classes,
        "layer": args.layer,
        "time_step": args.time_step,
    }


def loader_config(args, split: str, shuffle: bool) -> Dict[str, object]:
    family = str(getattr(args, "family", "cifar100"))
    return {
        "mode": family,
        "data_dir": args.data_dir,
        "split": split,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "shuffle": shuffle,
        "seed": args.seed,
    }


def build_model(root: Path, args, device: torch.device):
    model = build_cifar10_model(root, model_config(args)).to(device)
    checkpoint = load_checkpoint_if_available(model, args.checkpoint)
    return model, checkpoint


def freeze(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def enable_lutif_training(model, replacements) -> None:
    model.eval()
    for module in replacements.values():
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def stabilize_range(
    value_range,
    *,
    margin: float,
    name: str,
    kind: str,
    adjustments,
):
    lo, hi = map(float, value_range)
    span = hi - lo
    min_span = max(abs(lo), abs(hi), 1.0) * max(float(margin), 1e-3)
    if math.isfinite(lo + hi) and span >= min_span:
        return lo, hi
    if not math.isfinite(lo + hi):
        raise RuntimeError(f"non-finite {kind} range for {name}: {(lo, hi)}")
    center = 0.5 * (lo + hi)
    fixed = (center - 0.5 * min_span, center + 0.5 * min_span)
    adjustments.append(
        {
            "target": name,
            "kind": kind,
            "original": [lo, hi],
            "fixed": [fixed[0], fixed[1]],
            "reason": "degenerate_or_near_degenerate_calibration_range",
        }
    )
    return fixed


def build_ranges(clean_model, train_loader, device, args):
    targets = discover_lif_targets(clean_model)
    if not targets:
        raise RuntimeError("no MultiStepLIFNode targets were discovered")
    for module in targets.values():
        if hasattr(module, "store_v_seq"):
            module.store_v_seq = True
    seen, x_stats, v_stats = collect_lif_ranges(
        clean_model,
        train_loader,
        device,
        targets,
        args.calib_batches,
    )
    ranges = {}
    range_adjustments = []
    for name in targets:
        x_range = expanded_range(x_stats[name], args.range_margin)
        v_range = expanded_range(v_stats[name], args.range_margin)
        ranges[name] = {
            "x": stabilize_range(
                x_range,
                margin=args.range_margin,
                name=name,
                kind="input",
                adjustments=range_adjustments,
            ),
            "v": stabilize_range(
                v_range,
                margin=args.range_margin,
                name=name,
                kind="state",
                adjustments=range_adjustments,
            ),
        }
    return seen, ranges, range_adjustments


def top1(logits: torch.Tensor, target: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == target).sum().item())


def generic_module_summary(modules: Mapping[str, torch.nn.Module]) -> Dict[str, object]:
    per_module = {}
    entries = 0
    executions = 0
    clip_count = 0
    input_count = 0
    for name, module in modules.items():
        table_entries = int(module.table_entries()) if hasattr(module, "table_entries") else 0
        entries += table_entries
        executions += int(getattr(module, "executions", 0) > 0)
        clip_count += int(getattr(module, "clip_count", 0))
        input_count += int(getattr(module, "input_count", 0))
        per_module[name] = {
            "table_entries": table_entries,
            "executions": int(getattr(module, "executions", 0)),
        }
    return {
        "modules": len(modules),
        "modules_executed": executions,
        "table_entries": entries,
        "value_bytes_fp32": entries * 4,
        "value_kib_fp32": entries * 4.0 / 1024.0,
        "aggregate_clip_rate": clip_count / input_count if input_count else 0.0,
        "per_module": per_module,
    }


@torch.no_grad()
def evaluate_pair(
    teacher,
    candidate,
    loader,
    device,
    modules: Optional[Mapping[str, torch.nn.Module]],
    max_batches: Optional[int],
) -> Dict[str, float]:
    teacher.eval()
    candidate.eval()
    if modules is not None:
        reset_metrics(modules)
    correct_teacher = 0
    correct_candidate = 0
    total = 0
    candidate_ce = 0.0
    logit_mse = 0.0
    kl = 0.0
    for batch_index, (images, target) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        reset_snn(teacher)
        teacher_logits = teacher(images)
        reset_snn(candidate)
        candidate_logits = candidate(images)
        count = int(target.numel())
        correct_teacher += top1(teacher_logits, target)
        correct_candidate += top1(candidate_logits, target)
        candidate_ce += float(F.cross_entropy(candidate_logits.float(), target, reduction="sum").item())
        logit_mse += float(F.mse_loss(candidate_logits.float(), teacher_logits.float(), reduction="sum").item())
        kl += float(
            F.kl_div(
                F.log_softmax(candidate_logits.float(), dim=1),
                F.softmax(teacher_logits.float(), dim=1),
                reduction="sum",
            ).item()
        )
        total += count
    reset_snn(teacher)
    reset_snn(candidate)
    if total == 0:
        raise RuntimeError("evaluation loader produced no samples")
    return {
        "samples": total,
        "teacher_top1": 100.0 * correct_teacher / total,
        "candidate_top1": 100.0 * correct_candidate / total,
        "drop_top1": 100.0 * (correct_teacher - correct_candidate) / total,
        "candidate_ce": candidate_ce / total,
        "logit_mse_per_sample": logit_mse / total,
        "kl_teacher_to_candidate_per_sample": kl / total,
    }


def train_one_epoch(
    teacher,
    candidate,
    replacements,
    loader,
    optimizer,
    scaler,
    device,
    args,
    epoch: int,
) -> Dict[str, float]:
    teacher.eval()
    enable_lutif_training(candidate, replacements)
    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_delta = 0.0
    total_smooth = 0.0
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
            regularization = regularization_terms(replacements)
            loss = (
                ce
                + args.distill_weight * kd
                + args.delta_l2_weight * regularization["delta_l2"]
                + args.smoothness_weight * regularization["neighbor_smoothness"]
            )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite LUT-IF loss at epoch={epoch} batch={batch_index}: {loss.item()}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(trainable_parameters(replacements)), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        count = int(target.numel())
        samples += count
        batches += 1
        correct += top1(logits.detach(), target)
        total_loss += float(loss.detach().item()) * count
        total_ce += float(ce.detach().item()) * count
        total_kd += float(kd.detach().item()) * count
        total_delta += float(regularization["delta_l2"].detach().item()) * count
        total_smooth += float(regularization["neighbor_smoothness"].detach().item()) * count
        reset_snn(candidate)
    if samples == 0:
        raise RuntimeError("training loader produced no samples")
    return {
        "epoch": epoch,
        "batches": batches,
        "samples": samples,
        "train_top1": 100.0 * correct / samples,
        "loss": total_loss / samples,
        "ce": total_ce / samples,
        "kd": total_kd / samples,
        "delta_l2": total_delta / samples,
        "neighbor_smoothness": total_smooth / samples,
    }


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


def build_report(metrics: Mapping[str, object]) -> str:
    baseline = metrics["baseline"]
    final = metrics["final"]
    gate = metrics["gate"]
    formal_note = ""
    if gate.get("smoke_mode"):
        formal_note = (
            "\n\nSmoke mode uses a tiny subset and validates execution plumbing only; "
            "do not interpret the subset Acc@1 or formal full gate as paper evidence."
        )
    return f"""# LUT-IF Initial Proof-of-Concept\n\nStatus: **{gate['verdict']}**\n\n## Scope\n\n- QKFormer CIFAR-100, seed {metrics['protocol']['seed']}, T={metrics['protocol']['time_step']}\n- LIF-only intervention\n- {metrics['protocol']['state_bits']}-bit state and {metrics['protocol']['input_bits']}-bit input addresses\n- Frozen backbone; only LUT-IF transition tables and thresholds are trained{formal_note}\n\n## Baselines Before Training\n\n| Method | Acc@1 | Drop | Logit MSE/sample |\n|---|---:|---:|---:|\n| Post-hoc dense LUT | {baseline['posthoc']['candidate_top1']:.4f} | {baseline['posthoc']['drop_top1']:.4f} | {baseline['posthoc']['logit_mse_per_sample']:.6f} |\n| Quantized arithmetic LIF | {baseline['quantized_arithmetic']['candidate_top1']:.4f} | {baseline['quantized_arithmetic']['drop_top1']:.4f} | {baseline['quantized_arithmetic']['logit_mse_per_sample']:.6f} |\n| Trainable LUT-IF initialization | {baseline['lut_if_initial']['candidate_top1']:.4f} | {baseline['lut_if_initial']['drop_top1']:.4f} | {baseline['lut_if_initial']['logit_mse_per_sample']:.6f} |\n\n## Final LUT-IF Result\n\n- Acc@1: **{final['candidate_top1']:.4f}%**\n- Paired drop: **{final['drop_top1']:.4f} pp**\n- Logit MSE/sample: **{final['logit_mse_per_sample']:.6f}**\n- Initial proof-of-concept gate: **{gate['verdict']}**\n\n## Claim Boundary\n\nThis run can only test whether a trainable LUT-IF module provides an initial matched-precision advantage on the registered QKFormer setting. It does not establish compact factorization, broad transfer, or hardware efficiency.\n"""


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
    calibration_batches, ranges, range_adjustments = build_ranges(clean, train_loader, device, args)

    lut_if_model, lut_if_checkpoint = build_model(root, args, device)
    posthoc_model, posthoc_checkpoint = build_model(root, args, device)
    quant_model, quant_checkpoint = build_model(root, args, device)
    for model in (lut_if_model, posthoc_model, quant_model):
        freeze(model)

    lut_targets = discover_lif_targets(lut_if_model)
    posthoc_targets = discover_lif_targets(posthoc_model)
    quant_targets = discover_lif_targets(quant_model)
    lut_replacements = replace_with_dense_lutif(
        lut_if_model,
        lut_targets,
        ranges,
        state_bits=args.state_bits,
        input_bits=args.input_bits,
        surrogate_slope=args.surrogate_slope,
        learn_threshold=True,
    )
    posthoc_replacements = replace_lif_targets(
        posthoc_model,
        posthoc_targets,
        lif_lut_bits=args.state_bits,
        lif_ranges=ranges,
    )
    quant_replacements = replace_with_quantized_arithmetic(
        quant_model,
        quant_targets,
        ranges,
        state_bits=args.state_bits,
        input_bits=args.input_bits,
        surrogate_slope=args.surrogate_slope,
    )

    baseline_posthoc = evaluate_pair(
        clean, posthoc_model, validation_loader, device, posthoc_replacements, args.max_eval_batches
    )
    baseline_quant = evaluate_pair(
        clean, quant_model, validation_loader, device, quant_replacements, args.max_eval_batches
    )
    baseline_lut = evaluate_pair(
        clean, lut_if_model, validation_loader, device, lut_replacements, args.max_eval_batches
    )

    parameters = list(trainable_parameters(lut_replacements))
    if not parameters:
        raise RuntimeError("LUT-IF replacement produced no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    training_log = []
    best_top1 = -math.inf
    best_state = None
    best_eval = None
    for epoch in range(1, args.epochs + 1):
        train_row = train_one_epoch(
            clean,
            lut_if_model,
            lut_replacements,
            train_loader,
            optimizer,
            scaler,
            device,
            args,
            epoch,
        )
        validation = evaluate_pair(
            clean,
            lut_if_model,
            validation_loader,
            device,
            lut_replacements,
            args.max_eval_batches,
        )
        train_row.update({f"val_{key}": value for key, value in validation.items()})
        training_log.append(train_row)
        if validation["candidate_top1"] > best_top1:
            best_top1 = validation["candidate_top1"]
            best_eval = validation
            best_state = {
                name: {key: value.detach().cpu() for key, value in module.state_dict().items()}
                for name, module in lut_replacements.items()
            }

    if best_state is None or best_eval is None:
        raise RuntimeError("no best LUT-IF state was recorded")
    torch.save(
        {
            "protocol": vars(args),
            "replacements": best_state,
            "best_evaluation": best_eval,
        },
        result_dir / "lut_if_best.pt",
    )

    beats_posthoc = (
        best_eval["candidate_top1"] > baseline_posthoc["candidate_top1"]
        or best_eval["logit_mse_per_sample"] < baseline_posthoc["logit_mse_per_sample"]
    )
    beats_quant = (
        best_eval["candidate_top1"] > baseline_quant["candidate_top1"]
        or best_eval["logit_mse_per_sample"] < baseline_quant["logit_mse_per_sample"]
    )
    noninferior = best_eval["drop_top1"] <= args.max_drop
    lut_if_storage = metadata_summary(lut_replacements)
    posthoc_storage = generic_module_summary(posthoc_replacements)
    quant_storage = metadata_summary(quant_replacements)
    expected_targets = len(lut_replacements)
    all_targets_executed = bool(
        lut_if_storage["modules_executed"] == expected_targets
        and posthoc_storage["modules_executed"] == expected_targets
        and quant_storage["modules_executed"] == expected_targets
    )
    smoke_mode = args.max_eval_batches is not None
    smoke_pass = bool(all_targets_executed and len(training_log) > 0)
    gate_pass = bool(beats_posthoc and beats_quant and noninferior and all_targets_executed)

    metrics = {
        "experiment": "lut_if_initial_poc",
        "status": "completed",
        "protocol": {
            "seed": args.seed,
            "time_step": args.time_step,
            "state_bits": args.state_bits,
            "input_bits": args.input_bits,
            "epochs": args.epochs,
            "max_train_batches": args.max_train_batches,
            "max_eval_batches": args.max_eval_batches,
            "calibration_batches": calibration_batches,
            "range_adjustments": range_adjustments,
            "checkpoint": args.checkpoint,
            "data_dir": args.data_dir,
        },
        "checkpoint_load": {
            "clean": clean_checkpoint,
            "lut_if": lut_if_checkpoint,
            "posthoc": posthoc_checkpoint,
            "quantized_arithmetic": quant_checkpoint,
        },
        "baseline": {
            "posthoc": baseline_posthoc,
            "quantized_arithmetic": baseline_quant,
            "lut_if_initial": baseline_lut,
        },
        "training": training_log,
        "final": best_eval,
        "storage": {
            "lut_if": lut_if_storage,
            "posthoc": generic_module_summary(posthoc_replacements),
            "quantized_arithmetic": metadata_summary(quant_replacements),
        },
        "gate": {
            "beats_posthoc_on_top1_or_logit_mse": beats_posthoc,
            "beats_quantized_arithmetic_on_top1_or_logit_mse": beats_quant,
            "drop_within_initial_gate": noninferior,
            "all_requested_targets_executed": all_targets_executed,
            "smoke_mode": smoke_mode,
            "smoke_pass": smoke_pass,
            "expected_targets": expected_targets,
            "max_drop": args.max_drop,
            "formal_full_gate_pass": gate_pass,
            "pass": smoke_pass if smoke_mode else gate_pass,
            "verdict": (
                ("SMOKE-PASS" if smoke_pass else "SMOKE-FAIL")
                if smoke_mode
                else ("GO" if gate_pass else "NO-GO")
            ),
        },
        "claim_boundary": (
            "Initial matched-precision LUT-IF proof of concept only. No compactness, "
            "generality, or hardware claim is unlocked."
        ),
    }
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, default=str)
    save_csv(result_dir / "training_log.csv", training_log)
    summary_rows = []
    for name, row in {
        "posthoc": baseline_posthoc,
        "quantized_arithmetic": baseline_quant,
        "lut_if_initial": baseline_lut,
        "lut_if_final": best_eval,
    }.items():
        summary_rows.append({"method": name, **row})
    save_csv(result_dir / "summary.csv", summary_rows)
    (result_dir / "report.md").write_text(build_report(metrics), encoding="utf-8")
    return metrics


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Initial trainable LUT-IF proof of concept")
    p.add_argument("--root", default=".")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--result-dir", required=True)
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
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--surrogate-slope", type=float, default=2.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-drop", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
