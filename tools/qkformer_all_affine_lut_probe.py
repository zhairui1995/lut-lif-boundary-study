#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from tools.qkformer_lut_current_unit_probe import ScalarMeter, reset_snn
from tools.qkformer_lut_e2_replace import (
    accuracy,
    build_cifar10_model,
    build_loader,
    load_checkpoint_if_available,
)


def _target_category(name: str) -> Optional[str]:
    if name == "head":
        return "classifier"
    if name.startswith("patch_embed") and name.endswith("_conv"):
        return "patch"
    if name.endswith((".q_conv", ".k_conv", ".v_conv")):
        return "qkv"
    if name.endswith((".mlp1_conv", ".mlp2_conv")):
        return "mlp"
    if name.endswith(".proj_conv") and (".tssa." in name or ".ssa." in name):
        return "projection"
    return None


def discover_targets(model: torch.nn.Module, category: str) -> Dict[str, torch.nn.Module]:
    allowed = {"qkv", "mlp", "patch", "classifier", "projection", "all_affine"}
    if category not in allowed:
        raise ValueError(category)
    targets: Dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        target_category = _target_category(name)
        if target_category is None:
            continue
        if category == "all_affine" or target_category == category:
            if not isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                raise TypeError(f"unsupported target module {name}: {type(module)}")
            targets[name] = module
    if not targets:
        raise RuntimeError(f"no targets discovered for category={category}")
    return targets


class InputStats:
    def __init__(self) -> None:
        self.minimum = math.inf
        self.maximum = -math.inf
        self.count = 0
        self.integer_abs_sum = 0.0
        self.integer_abs_max = 0.0

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        value = x.detach().float()
        if value.numel() == 0:
            return
        self.minimum = min(self.minimum, float(value.min().item()))
        self.maximum = max(self.maximum, float(value.max().item()))
        integer_error = (value - value.round()).abs()
        self.integer_abs_sum += float(integer_error.sum().item())
        self.integer_abs_max = max(self.integer_abs_max, float(integer_error.max().item()))
        self.count += int(value.numel())

    def summary(self) -> Dict[str, float]:
        return {
            "minimum": 0.0 if self.count == 0 else self.minimum,
            "maximum": 0.0 if self.count == 0 else self.maximum,
            "count": float(self.count),
            "integer_mae": self.integer_abs_sum / self.count if self.count else 0.0,
            "integer_max_error": self.integer_abs_max,
        }


class ErrorStats:
    def __init__(self) -> None:
        self.sse = 0.0
        self.reference_energy = 0.0
        self.count = 0
        self.clip_count = 0
        self.input_count = 0
        self.quant_sse = 0.0

    @torch.no_grad()
    def update_output(self, original: torch.Tensor, replacement: torch.Tensor) -> None:
        diff = replacement.detach().float() - original.detach().float()
        ref = original.detach().float()
        self.sse += float((diff * diff).sum().item())
        self.reference_energy += float((ref * ref).sum().item())
        self.count += int(ref.numel())

    @torch.no_grad()
    def update_input(self, original: torch.Tensor, decoded: torch.Tensor, clipped: torch.Tensor) -> None:
        diff = decoded.detach().float() - original.detach().float()
        self.quant_sse += float((diff * diff).sum().item())
        self.clip_count += int(clipped.sum().item())
        self.input_count += int(original.numel())

    def summary(self) -> Dict[str, float]:
        mse = self.sse / self.count if self.count else 0.0
        nrmse = math.sqrt(self.sse / max(self.reference_energy, 1.0e-30))
        return {
            "output_mse": mse,
            "output_nrmse": nrmse,
            "clip_rate": self.clip_count / self.input_count if self.input_count else 0.0,
            "input_quant_mse": self.quant_sse / self.input_count if self.input_count else 0.0,
            "output_values": float(self.count),
            "input_values": float(self.input_count),
        }


class ScalarLevelLUT:
    def __init__(
        self,
        model: torch.nn.Module,
        targets: Dict[str, torch.nn.Module],
        integer_max: int,
        uniform_bits: int,
        input_chunk: int,
    ) -> None:
        self.model = model
        self.targets = targets
        self.integer_max = int(integer_max)
        self.uniform_bits = int(uniform_bits)
        self.input_chunk = int(input_chunk)
        self.enabled = False
        self.collecting = False
        self.input_stats: Dict[str, InputStats] = defaultdict(InputStats)
        self.error_stats: Dict[str, ErrorStats] = defaultdict(ErrorStats)
        self.ranges: Dict[str, Tuple[float, float]] = {}
        self.tables: Dict[Tuple[str, str, str], torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.executed: Dict[str, int] = defaultdict(int)
        self._register()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.tables.clear()

    def _register(self) -> None:
        for name, module in self.targets.items():
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, inputs, output):
            if not inputs:
                return None
            x = inputs[0]
            if self.collecting:
                self.input_stats[name].update(x)
                return None
            if not self.enabled:
                return None
            replacement = self._replace(name, x, output)
            self.error_stats[name].update_output(output, replacement)
            self.executed[name] += 1
            return replacement

        return hook

    @staticmethod
    def _continuous_target(name: str) -> bool:
        return name == "patch_embed1.proj_conv" or name == "head"

    def finalize_calibration(self) -> None:
        for name in self.targets:
            summary = self.input_stats[name].summary()
            self.ranges[name] = (float(summary["minimum"]), float(summary["maximum"]))
        self.tables.clear()

    def reset_eval_metrics(self) -> None:
        self.error_stats = defaultdict(ErrorStats)
        self.executed = defaultdict(int)

    def set_passthrough(self) -> None:
        self.enabled = False
        self.collecting = False

    def set_collecting(self) -> None:
        self.enabled = False
        self.collecting = True

    def set_enabled(self) -> None:
        self.collecting = False
        self.enabled = True
        self.reset_eval_metrics()

    def _quantize(self, name: str, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = x.detach().float()
        if self._continuous_target(name):
            levels = (1 << self.uniform_bits) - 1
            lo, hi = self.ranges[name]
            if hi <= lo:
                decoded = torch.full_like(value, lo)
                index = torch.zeros_like(value, dtype=torch.long)
                clipped = torch.zeros_like(value, dtype=torch.bool)
            else:
                clipped = (value < lo) | (value > hi)
                scaled = (value.clamp(lo, hi) - lo) * (levels / (hi - lo))
                index = scaled.round().long().clamp(0, levels)
                decoded = lo + index.float() * ((hi - lo) / levels)
        else:
            clipped = (value < 0.0) | (value > float(self.integer_max))
            index = value.round().long().clamp(0, self.integer_max)
            decoded = index.float()
        self.error_stats[name].update_input(value, decoded, clipped)
        return index, decoded, clipped

    def _decoded_levels(self, name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._continuous_target(name):
            count = 1 << self.uniform_bits
            lo, hi = self.ranges[name]
            if count <= 1 or hi <= lo:
                return torch.full((count,), lo, device=device, dtype=dtype)
            return torch.linspace(lo, hi, count, device=device, dtype=dtype)
        return torch.arange(self.integer_max + 1, device=device, dtype=dtype)

    def _flatten_weight(self, module: torch.nn.Module) -> torch.Tensor:
        if isinstance(module, torch.nn.Linear):
            return module.weight.detach()
        return module.weight.detach().reshape(module.out_channels, -1)

    def _table(self, name: str, module: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (name, str(device), str(dtype))
        table = self.tables.get(key)
        if table is not None:
            return table
        weight = self._flatten_weight(module).to(device=device, dtype=dtype)
        levels = self._decoded_levels(name, device, dtype)
        table = weight.transpose(0, 1)[:, :, None] * levels[None, None, :]
        table = table.permute(0, 2, 1).contiguous()
        self.tables[key] = table
        return table

    @staticmethod
    def _columns(
        module: torch.nn.Module,
        x: torch.Tensor,
        zero_index: int,
    ) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        if isinstance(module, torch.nn.Linear):
            shape = tuple(x.shape[:-1])
            return x.reshape(-1, x.shape[-1]).unsqueeze(-1), shape
        if isinstance(module, torch.nn.Conv1d):
            if module.kernel_size != (1,) or module.stride != (1,) or module.padding != (0,) or module.dilation != (1,):
                raise ValueError(f"unsupported Conv1d geometry: {module}")
            return x, (x.shape[0], module.out_channels, x.shape[-1])
        if isinstance(module, torch.nn.Conv2d):
            original_h, original_w = x.shape[-2:]
            padding_h, padding_w = module.padding
            if padding_h or padding_w:
                x = F.pad(
                    x,
                    (padding_w, padding_w, padding_h, padding_h),
                    mode="constant",
                    value=int(zero_index),
                )
            columns = F.unfold(
                x.float(),
                kernel_size=module.kernel_size,
                dilation=module.dilation,
                padding=0,
                stride=module.stride,
            ).long()
            h_out = (
                (original_h + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1)
                // module.stride[0]
                + 1
            )
            w_out = (
                (original_w + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1)
                // module.stride[1]
                + 1
            )
            return columns, (x.shape[0], module.out_channels, h_out, w_out)
        raise TypeError(type(module))

    def _replace(self, name: str, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        module = self.targets[name]
        index, _decoded, _clipped = self._quantize(name, x)
        if self._continuous_target(name):
            levels = (1 << self.uniform_bits) - 1
            lo, hi = self.ranges[name]
            zero_index = 0 if hi <= lo else int(round((0.0 - lo) * levels / (hi - lo)))
            zero_index = max(0, min(levels, zero_index))
        else:
            zero_index = 0
        columns, output_shape = self._columns(module, index, zero_index)
        columns = columns.long()
        table = self._table(name, module, output.device, output.dtype)
        batch, features, positions = columns.shape
        result = torch.zeros(
            (batch, positions, table.shape[-1]),
            device=output.device,
            dtype=output.dtype,
        )
        if module.bias is not None:
            result += module.bias.detach().to(device=output.device, dtype=output.dtype).view(1, 1, -1)
        for start in range(0, features, self.input_chunk):
            end = min(features, start + self.input_chunk)
            feature_index = torch.arange(start, end, device=output.device).view(1, -1, 1)
            feature_index = feature_index.expand(batch, -1, positions)
            contribution = table[feature_index, columns[:, start:end, :]]
            result += contribution.sum(dim=1)
        if isinstance(module, torch.nn.Linear):
            return result.squeeze(1).reshape(*output_shape, table.shape[-1])
        if isinstance(module, torch.nn.Conv1d):
            return result.permute(0, 2, 1).reshape(output_shape)
        return result.permute(0, 2, 1).reshape(output_shape)

    def table_entries(self) -> int:
        total = 0
        for name, module in self.targets.items():
            levels = (1 << self.uniform_bits) if self._continuous_target(name) else self.integer_max + 1
            weight = self._flatten_weight(module)
            total += int(weight.shape[0] * weight.shape[1] * levels)
        return total

    def original_weight_values(self) -> int:
        return sum(int(self._flatten_weight(module).numel()) for module in self.targets.values())

    def summary(self) -> Dict[str, object]:
        per_target = {}
        total_sse = 0.0
        total_energy = 0.0
        total_outputs = 0
        total_clips = 0
        total_inputs = 0
        total_quant_sse = 0.0
        for name in self.targets:
            metric = self.error_stats[name]
            row = {
                **self.input_stats[name].summary(),
                **metric.summary(),
                "executions": self.executed[name],
                "continuous_quantizer": self._continuous_target(name),
            }
            per_target[name] = row
            total_sse += metric.sse
            total_energy += metric.reference_energy
            total_outputs += metric.count
            total_clips += metric.clip_count
            total_inputs += metric.input_count
            total_quant_sse += metric.quant_sse
        entries = self.table_entries()
        weights = self.original_weight_values()
        return {
            "output_mse": total_sse / total_outputs if total_outputs else 0.0,
            "output_nrmse": math.sqrt(total_sse / max(total_energy, 1.0e-30)),
            "clip_rate": total_clips / total_inputs if total_inputs else 0.0,
            "input_quant_mse": total_quant_sse / total_inputs if total_inputs else 0.0,
            "table_scalar_entries": entries,
            "table_fp32_kib": entries * 4.0 / 1024.0,
            "original_weight_values": weights,
            "table_to_weight_value_ratio": entries / weights if weights else 0.0,
            "targets_requested": len(self.targets),
            "targets_executed": sum(1 for name in self.targets if self.executed[name] > 0),
            "per_target": per_target,
        }


@torch.no_grad()
def calibrate(model, loader, device, hook: ScalarLevelLUT, batches: int) -> int:
    model.eval()
    hook.set_collecting()
    seen = 0
    for images, _target in loader:
        if seen >= batches:
            break
        reset_snn(model)
        model(images.to(device, non_blocking=True))
        seen += 1
    reset_snn(model)
    hook.set_passthrough()
    hook.finalize_calibration()
    return seen


@torch.no_grad()
def evaluate(model, loader, device, hook: ScalarLevelLUT, max_batches: Optional[int]) -> Dict[str, object]:
    model.eval()
    clean_top1 = ScalarMeter()
    injected_top1 = ScalarMeter()
    kl = ScalarMeter()
    logit_mse = ScalarMeter()
    hook.reset_eval_metrics()
    for batch_index, (images, target) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        hook.set_passthrough()
        reset_snn(model)
        clean = model(images)
        hook.enabled = True
        reset_snn(model)
        injected = model(images)
        hook.enabled = False
        count = int(target.numel())
        clean_top1.update(accuracy(clean, target, topk=(1,))[0], count)
        injected_top1.update(accuracy(injected, target, topk=(1,))[0], count)
        kl.update(
            float(
                F.kl_div(
                    F.log_softmax(injected.float(), dim=1),
                    F.softmax(clean.float(), dim=1),
                    reduction="batchmean",
                ).item()
            ),
            count,
        )
        logit_mse.update(float(F.mse_loss(injected.float(), clean.float()).item()), count)
    reset_snn(model)
    local = hook.summary()
    return {
        "clean_top1": clean_top1.mean,
        "injected_top1": injected_top1.mean,
        "drop_top1": clean_top1.mean - injected_top1.mean,
        "kl_clean_to_injected": kl.mean,
        "logit_mse": logit_mse.mean,
        **local,
    }


def write_csv(path: Path, row: Dict[str, object]) -> None:
    scalar = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar.keys()))
        writer.writeheader()
        writer.writerow(scalar)


def run(args) -> Dict[str, object]:
    root = Path(args.root).resolve()
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.local_cuda_index}" if torch.cuda.is_available() else "cpu")
    model_cfg = {
        "family": args.family,
        "img_size": 32,
        "patch_size": 4,
        "dim": args.dim,
        "num_heads": 8,
        "mlp_ratio": 4,
        "in_channels": 3,
        "num_classes": 100 if args.family == "cifar100" else 10,
        "layer": args.layer,
        "time_step": args.time_step,
    }
    model = build_cifar10_model(root, model_cfg).to(device)
    checkpoint = load_checkpoint_if_available(model, args.checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = discover_targets(model, args.category)
    hook = ScalarLevelLUT(
        model,
        targets,
        integer_max=args.integer_max,
        uniform_bits=args.uniform_bits,
        input_chunk=args.input_chunk,
    )
    data_cfg = {
        "mode": args.family,
        "data_dir": args.data_dir,
        "split": "train",
        "batch_size": args.batch_size,
        "workers": args.workers,
        "shuffle": True,
        "seed": args.seed,
    }
    validation_cfg = dict(data_cfg)
    validation_cfg.update({"split": "validation", "shuffle": False})
    train_loader = build_loader(data_cfg, device)
    validation_loader = build_loader(validation_cfg, device)
    calibration_batches = calibrate(model, train_loader, device, hook, args.calib_batches)
    metrics = evaluate(model, validation_loader, device, hook, args.max_eval_batches)
    hook.close()
    gate = {
        "clean_baseline_pass": abs(metrics["clean_top1"] - args.expected_clean_top1) <= args.clean_tolerance,
        "drop_top1_pass": metrics["drop_top1"] <= args.max_drop,
        "output_nrmse_pass": metrics["output_nrmse"] <= args.max_nrmse,
        "clip_rate_pass": metrics["clip_rate"] <= args.max_clip_rate,
        "all_targets_executed": metrics["targets_executed"] == metrics["targets_requested"],
        "finite": all(
            math.isfinite(float(metrics[key]))
            for key in ("clean_top1", "injected_top1", "output_nrmse", "clip_rate")
        ),
    }
    gate["pass"] = all(bool(value) for value in gate.values())
    payload = {
        "experiment": "qkformer_all_affine_lut_probe",
        "claim": (
            "LUT-equivalent fidelity audit for frozen learned affine operators; "
            "original operators still execute before hook substitution; BN/LIF/pooling/"
            "residual/attention matrix products remain unchanged"
        ),
        "checkpoint": checkpoint,
        "category": args.category,
        "targets": list(targets),
        "calibration_batches": calibration_batches,
        "args": vars(args),
        "metrics": metrics,
        "gate": gate,
    }
    (result_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(result_dir / "summary.csv", {**metrics, **{f"gate_{key}": value for key, value in gate.items()}})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--family", choices=("cifar10", "cifar100"), default="cifar100")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--category", choices=("qkv", "mlp", "patch", "classifier", "projection", "all_affine"), required=True)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--time-step", type=int, choices=(1, 4), default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--calib-batches", type=int, default=32)
    parser.add_argument("--max-eval-batches", type=int, default=100000)
    parser.add_argument("--integer-max", type=int, default=7)
    parser.add_argument("--uniform-bits", type=int, default=8)
    parser.add_argument("--input-chunk", type=int, default=4)
    parser.add_argument("--max-drop", type=float, default=0.25)
    parser.add_argument("--max-nrmse", type=float, default=0.01)
    parser.add_argument("--max-clip-rate", type=float, default=0.0001)
    parser.add_argument("--expected-clean-top1", type=float, default=77.78)
    parser.add_argument("--clean-tolerance", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-cuda-index", type=int, default=0)
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "result_dir": str(Path(args.result_dir).resolve()),
                "category": args.category,
                "gate": payload["gate"],
                "drop_top1": payload["metrics"]["drop_top1"],
                "output_nrmse": payload["metrics"]["output_nrmse"],
                "clip_rate": payload["metrics"]["clip_rate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
