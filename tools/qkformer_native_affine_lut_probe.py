#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from tools.qkformer_all_affine_lut_probe import InputStats, discover_targets
from tools.qkformer_lut_current_unit_probe import ScalarMeter, reset_snn
from tools.qkformer_lut_e2_replace import (
    accuracy,
    build_cifar10_model,
    build_loader,
    load_checkpoint_if_available,
)


class NativeMultiStepLIF(torch.nn.Module):
    """Drop-in native implementation of eval-time SpikingJelly MultiStepLIFNode."""

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = True,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
    ) -> None:
        super().__init__()
        self.tau = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = None if v_reset is None else float(v_reset)
        self.v = 0.0
        self.v_seq: Optional[torch.Tensor] = None
        self.executions = 0

    def reset(self) -> None:
        self.v = 0.0
        self.v_seq = None

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() <= 1:
            raise ValueError("NativeMultiStepLIF expects [T, ...] input")
        self.executions += 1
        if isinstance(self.v, float):
            v = torch.zeros_like(x_seq[0])
            if self.v != 0.0:
                v.fill_(float(self.v))
        else:
            v = self.v.to(device=x_seq.device, dtype=x_seq.dtype)
        spikes = []
        states = []
        for t in range(x_seq.shape[0]):
            x = x_seq[t]
            if self.decay_input:
                if self.v_reset is None or self.v_reset == 0.0:
                    v = v + (x - v) / self.tau
                else:
                    v = v + (x - (v - self.v_reset)) / self.tau
            else:
                if self.v_reset is None or self.v_reset == 0.0:
                    v = v * (1.0 - 1.0 / self.tau) + x
                else:
                    v = v - (v - self.v_reset) / self.tau + x
            spike = (v - self.v_threshold >= 0).to(x)
            if self.v_reset is None:
                v = v - spike * self.v_threshold
            else:
                v = (1.0 - spike) * v + spike * self.v_reset
            spikes.append(spike.unsqueeze(0))
            states.append(v.unsqueeze(0))
        self.v = states[-1].squeeze(0).detach().clone()
        self.v_seq = torch.cat(states, dim=0)
        return torch.cat(spikes, dim=0)


class QuantizedTransitionLUT(torch.nn.Module):
    """Finite per-element LUT for (membrane state, input) -> (spike, next state)."""

    def __init__(
        self,
        tau: float,
        decay_input: bool,
        v_threshold: float,
        v_reset: Optional[float],
        x_range: Tuple[float, float],
        v_range: Tuple[float, float],
        bits: int,
        x_integer_range: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.tau = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = None if v_reset is None else float(v_reset)
        self.x_range = (float(x_range[0]), float(x_range[1]))
        self.v_range = (float(v_range[0]), float(v_range[1]))
        self.x_integer_range = x_integer_range
        self.bits = int(bits)
        if self.bits < 1 or self.bits > 10:
            raise ValueError(f"LIF transition LUT bits must be in [1, 10], got {self.bits}")
        self.v_levels = 1 << self.bits
        self.x_levels = (
            int(x_integer_range[1] - x_integer_range[0] + 1)
            if x_integer_range is not None
            else 1 << self.bits
        )
        self.v = 0.0
        self.v_seq: Optional[torch.Tensor] = None
        self.executions = 0
        self.clip_count = 0
        self.input_count = 0
        self.quant_sse = 0.0
        self._table_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    def reset(self) -> None:
        self.v = 0.0
        self.v_seq = None

    def _quantize(
        self,
        value: torch.Tensor,
        value_range: Tuple[float, float],
        levels: int,
        integer_range: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        fp32 = value.detach().float()
        if integer_range is not None:
            lo_int, hi_int = integer_range
            clipped = (fp32 < float(lo_int)) | (fp32 > float(hi_int))
            index = fp32.round().long().clamp(lo_int, hi_int) - lo_int
            decoded = (index + lo_int).float()
            diff = decoded - fp32
            self.clip_count += int(clipped.sum().item())
            self.input_count += int(fp32.numel())
            self.quant_sse += float((diff * diff).sum().item())
            return index
        lo, hi = value_range
        if hi <= lo:
            index = torch.zeros_like(fp32, dtype=torch.long)
            decoded = torch.full_like(fp32, lo)
            clipped = fp32 != lo
        else:
            clipped = (fp32 < lo) | (fp32 > hi)
            scaled = (fp32.clamp(lo, hi) - lo) * ((levels - 1) / (hi - lo))
            index = scaled.round().long().clamp(0, levels - 1)
            decoded = lo + index.float() * ((hi - lo) / (levels - 1))
        diff = decoded - fp32
        self.clip_count += int(clipped.sum().item())
        self.input_count += int(fp32.numel())
        self.quant_sse += float((diff * diff).sum().item())
        return index

    def _decoded_levels(
        self,
        value_range: Tuple[float, float],
        levels: int,
        device: torch.device,
        dtype: torch.dtype,
        integer_range: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if integer_range is not None:
            return torch.arange(
                integer_range[0],
                integer_range[1] + 1,
                device=device,
                dtype=dtype,
            )
        lo, hi = value_range
        if hi <= lo:
            return torch.full((levels,), lo, device=device, dtype=dtype)
        return torch.linspace(lo, hi, levels, device=device, dtype=dtype)

    def _tables(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), str(dtype))
        cached = self._table_cache.get(key)
        if cached is not None:
            return cached
        x = self._decoded_levels(
            self.x_range,
            self.x_levels,
            device,
            dtype,
            integer_range=self.x_integer_range,
        ).view(1, -1)
        v = self._decoded_levels(
            self.v_range,
            self.v_levels,
            device,
            dtype,
        ).view(-1, 1)
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.0:
                charged = v + (x - v) / self.tau
            else:
                charged = v + (x - (v - self.v_reset)) / self.tau
        else:
            if self.v_reset is None or self.v_reset == 0.0:
                charged = v * (1.0 - 1.0 / self.tau) + x
            else:
                charged = v - (v - self.v_reset) / self.tau + x
        spike = (charged - self.v_threshold >= 0).to(dtype)
        if self.v_reset is None:
            next_v = charged - spike * self.v_threshold
        else:
            next_v = (1.0 - spike) * charged + spike * self.v_reset
        cached = (spike.reshape(-1), next_v.reshape(-1))
        self._table_cache[key] = cached
        return cached

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() <= 1:
            raise ValueError("QuantizedTransitionLUT expects [T, ...] input")
        self.executions += 1
        if isinstance(self.v, float):
            v = torch.zeros_like(x_seq[0])
            if self.v != 0.0:
                v.fill_(float(self.v))
        else:
            v = self.v.to(device=x_seq.device, dtype=x_seq.dtype)
        spike_table, state_table = self._tables(x_seq.device, x_seq.dtype)
        spikes = []
        states = []
        for t in range(x_seq.shape[0]):
            x_index = self._quantize(
                x_seq[t],
                self.x_range,
                self.x_levels,
                integer_range=self.x_integer_range,
            )
            v_index = self._quantize(v, self.v_range, self.v_levels)
            flat_index = v_index * self.x_levels + x_index
            spike = spike_table[flat_index]
            v = state_table[flat_index]
            spikes.append(spike.unsqueeze(0))
            states.append(v.unsqueeze(0))
        self.v = states[-1].squeeze(0).detach().clone()
        self.v_seq = torch.cat(states, dim=0)
        return torch.cat(spikes, dim=0)

    def table_entries(self) -> int:
        return 2 * self.x_levels * self.v_levels

    def summary(self) -> Dict[str, float]:
        return {
            "executions": float(self.executions),
            "x_levels": float(self.x_levels),
            "v_levels": float(self.v_levels),
            "integer_x": float(self.x_integer_range is not None),
            "clip_rate": self.clip_count / self.input_count if self.input_count else 0.0,
            "quant_mse": self.quant_sse / self.input_count if self.input_count else 0.0,
            "quantized_values": float(self.input_count),
            "table_scalar_entries": float(self.table_entries()),
        }


def _continuous_target(name: str) -> bool:
    return name == "patch_embed1.proj_conv" or name == "head"


class NativeScalarLevelLUT(torch.nn.Module):
    """LUT-native replacement for a frozen Conv1d/Conv2d/Linear module.

    The original affine module is not retained as a child module. The forward
    path performs quantize -> lookup contribution -> accumulation, so any
    original Conv/Linear target calls during evaluation must be zero by graph
    construction and by the explicit post-replacement call counter.
    """

    def __init__(
        self,
        name: str,
        module: torch.nn.Module,
        input_range: Tuple[float, float],
        integer_max: int,
        uniform_bits: int,
        input_chunk: int,
        folded_weight: Optional[torch.Tensor] = None,
        folded_bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if not isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
            raise TypeError(f"unsupported native LUT source for {name}: {type(module)}")
        self.name = name
        self.kind = module.__class__.__name__
        self.integer_max = int(integer_max)
        self.uniform_bits = int(uniform_bits)
        self.input_chunk = int(input_chunk)
        self.input_range = (float(input_range[0]), float(input_range[1]))
        self.executions = 0
        self.clip_count = 0
        self.input_count = 0
        self.input_quant_sse = 0.0
        self._table_cache: Dict[Tuple[str, str], torch.Tensor] = {}

        if isinstance(module, torch.nn.Linear):
            self.in_features = int(module.in_features)
            self.out_features = int(module.out_features)
            weight = module.weight.detach() if folded_weight is None else folded_weight.detach()
            self.register_buffer("weight_flat", weight.clone())
        else:
            self.in_channels = int(module.in_channels)
            self.out_channels = int(module.out_channels)
            self.kernel_size = tuple(int(v) for v in module.kernel_size)
            self.stride = tuple(int(v) for v in module.stride)
            self.padding = tuple(int(v) for v in module.padding)
            self.dilation = tuple(int(v) for v in module.dilation)
            self.groups = int(module.groups)
            if self.groups != 1:
                raise ValueError(f"grouped convolutions are not supported: {name}")
            weight = module.weight.detach() if folded_weight is None else folded_weight.detach()
            self.register_buffer("weight_flat", weight.reshape(module.out_channels, -1).clone())

        if folded_bias is not None:
            self.register_buffer("bias_value", folded_bias.detach().clone())
        elif module.bias is None:
            self.register_buffer("bias_value", torch.empty(0))
        else:
            self.register_buffer("bias_value", module.bias.detach().clone())

    def extra_repr(self) -> str:
        return (
            f"name={self.name}, kind={self.kind}, integer_max={self.integer_max}, "
            f"uniform_bits={self.uniform_bits}, input_chunk={self.input_chunk}"
        )

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        value = x.detach().float()
        if _continuous_target(self.name):
            levels = (1 << self.uniform_bits) - 1
            lo, hi = self.input_range
            if hi <= lo:
                index = torch.zeros_like(value, dtype=torch.long)
                decoded = torch.full_like(value, lo)
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
        diff = decoded - value
        self.input_quant_sse += float((diff * diff).sum().item())
        self.clip_count += int(clipped.sum().item())
        self.input_count += int(value.numel())
        return index

    def _decoded_levels(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if _continuous_target(self.name):
            count = 1 << self.uniform_bits
            lo, hi = self.input_range
            if count <= 1 or hi <= lo:
                return torch.full((count,), lo, device=device, dtype=dtype)
            return torch.linspace(lo, hi, count, device=device, dtype=dtype)
        return torch.arange(self.integer_max + 1, device=device, dtype=dtype)

    def _table(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), str(dtype))
        cached = self._table_cache.get(key)
        if cached is not None:
            return cached
        weight = self.weight_flat.to(device=device, dtype=dtype)
        levels = self._decoded_levels(device, dtype)
        table = weight.transpose(0, 1)[:, :, None] * levels[None, None, :]
        table = table.permute(0, 2, 1).contiguous()
        self._table_cache[key] = table
        return table

    def _zero_index(self) -> int:
        if not _continuous_target(self.name):
            return 0
        levels = (1 << self.uniform_bits) - 1
        lo, hi = self.input_range
        if hi <= lo:
            return 0
        return max(0, min(levels, int(round((0.0 - lo) * levels / (hi - lo)))))

    def _columns(self, index: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        zero_index = self._zero_index()
        if self.kind == "Linear":
            shape = tuple(index.shape[:-1])
            return index.reshape(-1, index.shape[-1]).unsqueeze(-1), shape
        if self.kind == "Conv1d":
            if self.kernel_size != (1,) or self.stride != (1,) or self.padding != (0,) or self.dilation != (1,):
                raise ValueError(f"unsupported Conv1d geometry for {self.name}")
            return index, (index.shape[0], self.out_channels, index.shape[-1])
        if self.kind == "Conv2d":
            original_h, original_w = index.shape[-2:]
            padding_h, padding_w = self.padding
            if padding_h or padding_w:
                index = F.pad(
                    index,
                    (padding_w, padding_w, padding_h, padding_h),
                    mode="constant",
                    value=zero_index,
                )
            columns = F.unfold(
                index.float(),
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=0,
                stride=self.stride,
            ).long()
            h_out = (
                (original_h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
                // self.stride[0]
                + 1
            )
            w_out = (
                (original_w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
                // self.stride[1]
                + 1
            )
            return columns, (index.shape[0], self.out_channels, h_out, w_out)
        raise ValueError(f"unsupported kind={self.kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.executions += 1
        index = self._quantize(x)
        columns, output_shape = self._columns(index)
        columns = columns.long()
        table = self._table(x.device, x.dtype)
        batch, features, positions = columns.shape
        result = torch.zeros((batch, positions, table.shape[-1]), device=x.device, dtype=x.dtype)
        if self.bias_value.numel():
            result += self.bias_value.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        for start in range(0, features, self.input_chunk):
            end = min(features, start + self.input_chunk)
            feature_index = torch.arange(start, end, device=x.device).view(1, -1, 1)
            feature_index = feature_index.expand(batch, -1, positions)
            contribution = table[feature_index, columns[:, start:end, :]]
            result += contribution.sum(dim=1)
        if self.kind == "Linear":
            return result.squeeze(1).reshape(*output_shape, table.shape[-1])
        if self.kind == "Conv1d":
            return result.permute(0, 2, 1).reshape(output_shape)
        return result.permute(0, 2, 1).reshape(output_shape)

    def table_entries(self) -> int:
        levels = (1 << self.uniform_bits) if _continuous_target(self.name) else self.integer_max + 1
        return int(self.weight_flat.numel() * levels)

    def summary(self) -> Dict[str, float]:
        return {
            "executions": float(self.executions),
            "clip_rate": self.clip_count / self.input_count if self.input_count else 0.0,
            "input_quant_mse": self.input_quant_sse / self.input_count if self.input_count else 0.0,
            "input_values": float(self.input_count),
            "table_scalar_entries": float(self.table_entries()),
        }


def collect_input_ranges(model, loader, device, targets: Dict[str, torch.nn.Module], batches: int) -> Tuple[int, Dict[str, InputStats]]:
    stats: Dict[str, InputStats] = defaultdict(InputStats)
    handles = []
    for name, module in targets.items():
        def make_hook(target_name: str):
            def hook(_module, inputs):
                if inputs:
                    stats[target_name].update(inputs[0])
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
    return seen, stats


def collect_lif_ranges(
    model,
    loader,
    device,
    targets: Dict[str, torch.nn.Module],
    batches: int,
) -> Tuple[int, Dict[str, InputStats], Dict[str, InputStats]]:
    x_stats: Dict[str, InputStats] = defaultdict(InputStats)
    v_stats: Dict[str, InputStats] = defaultdict(InputStats)
    handles = []
    for name, module in targets.items():
        def make_pre_hook(target_name: str):
            def hook(_module, inputs):
                if inputs:
                    x_stats[target_name].update(inputs[0])
            return hook

        def make_post_hook(target_name: str):
            def hook(current_module, _inputs, _output):
                v_seq = getattr(current_module, "v_seq", None)
                if not isinstance(v_seq, torch.Tensor):
                    raise RuntimeError(f"missing v_seq while calibrating {target_name}")
                v_stats[target_name].update(torch.zeros_like(v_seq[0]))
                if v_seq.shape[0] > 1:
                    v_stats[target_name].update(v_seq[:-1])
            return hook

        handles.append(module.register_forward_pre_hook(make_pre_hook(name)))
        handles.append(module.register_forward_hook(make_post_hook(name)))
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
    return seen, x_stats, v_stats


def expanded_range(stats: InputStats, margin: float) -> Tuple[float, float]:
    summary = stats.summary()
    lo = float(summary["minimum"])
    hi = float(summary["maximum"])
    span = hi - lo
    if span == 0.0:
        return lo, hi
    pad = max(abs(lo), abs(hi), span, 1.0) * float(margin)
    return lo - pad, hi + pad


def set_submodule(root: torch.nn.Module, name: str, module: torch.nn.Module) -> None:
    if "." not in name:
        setattr(root, name, module)
        return
    parent_name, child_name = name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    setattr(parent, child_name, module)


def paired_bn_name(name: str) -> Optional[str]:
    if not name.endswith("_conv"):
        return None
    return name[:-5] + "_bn"


def lookup_paired_bn(model: torch.nn.Module, name: str) -> Optional[torch.nn.Module]:
    bn_name = paired_bn_name(name)
    if bn_name is None:
        return None
    try:
        bn = model.get_submodule(bn_name)
    except AttributeError:
        return None
    if isinstance(bn, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        return bn
    return None


def fold_bn_parameters(
    module: torch.nn.Module,
    bn: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
        raise TypeError(f"BN folding only supports conv modules, got {type(module)}")
    if not isinstance(bn, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        raise TypeError(f"unsupported BN type: {type(bn)}")
    weight = module.weight.detach().clone()
    if module.bias is None:
        bias = torch.zeros(weight.shape[0], dtype=weight.dtype, device=weight.device)
    else:
        bias = module.bias.detach().clone()
    running_mean = bn.running_mean.detach().to(device=weight.device, dtype=weight.dtype)
    running_var = bn.running_var.detach().to(device=weight.device, dtype=weight.dtype)
    if bn.affine:
        gamma = bn.weight.detach().to(device=weight.device, dtype=weight.dtype)
        beta = bn.bias.detach().to(device=weight.device, dtype=weight.dtype)
    else:
        gamma = torch.ones_like(running_mean)
        beta = torch.zeros_like(running_mean)
    scale = gamma / torch.sqrt(running_var + float(bn.eps))
    folded_weight = weight * scale.reshape(-1, *([1] * (weight.dim() - 1)))
    folded_bias = (bias - running_mean) * scale + beta
    return folded_weight, folded_bias


def attach_remaining_original_counter(model: torch.nn.Module, target_names):
    remaining = 0
    counter = {"calls": 0}
    handles = []
    target_set = set(target_names)
    for name, module in model.named_modules():
        if name not in target_set:
            continue
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
            remaining += 1
            def hook(_module, _inputs, _output):
                counter["calls"] += 1
            handles.append(module.register_forward_hook(hook))
    return remaining, counter, handles


def attach_remaining_bn_counter(model: torch.nn.Module, bn_names):
    remaining = 0
    counter = {"calls": 0}
    handles = []
    target_set = set(bn_names)
    for name, module in model.named_modules():
        if name not in target_set:
            continue
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            remaining += 1
            def hook(_module, _inputs, _output):
                counter["calls"] += 1
            handles.append(module.register_forward_hook(hook))
    return remaining, counter, handles


def discover_lif_targets(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    targets = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ == "MultiStepLIFNode":
            targets[name] = module
    return targets


def attach_remaining_lif_counter(model: torch.nn.Module, lif_names):
    remaining = 0
    counter = {"calls": 0}
    handles = []
    target_set = set(lif_names)
    for name, module in model.named_modules():
        if name not in target_set:
            continue
        if module.__class__.__name__ == "MultiStepLIFNode":
            remaining += 1
            def hook(_module, _inputs, _output):
                counter["calls"] += 1
            handles.append(module.register_forward_hook(hook))
    return remaining, counter, handles


def replace_lif_targets(
    model: torch.nn.Module,
    lif_targets: Dict[str, torch.nn.Module],
    lif_lut_bits: int,
    lif_ranges: Dict[str, Dict[str, object]],
) -> Dict[str, torch.nn.Module]:
    replacements = {}
    for name, module in list(lif_targets.items()):
        common = {
            "tau": float(getattr(module, "tau")),
            "decay_input": bool(getattr(module, "decay_input")),
            "v_threshold": float(getattr(module, "v_threshold")),
            "v_reset": getattr(module, "v_reset"),
        }
        if lif_lut_bits > 0:
            replacement = QuantizedTransitionLUT(
                **common,
                x_range=lif_ranges[name]["x"],
                v_range=lif_ranges[name]["v"],
                bits=lif_lut_bits,
                x_integer_range=lif_ranges[name].get("x_integer"),
            )
        else:
            replacement = NativeMultiStepLIF(**common)
        set_submodule(model, name, replacement)
        replacements[name] = replacement
    return replacements


def replace_targets(
    model: torch.nn.Module,
    targets: Dict[str, torch.nn.Module],
    ranges: Dict[str, Tuple[float, float]],
    integer_max: int,
    uniform_bits: int,
    input_chunk: int,
    fold_bn: bool,
) -> Dict[str, NativeScalarLevelLUT]:
    wrappers: Dict[str, NativeScalarLevelLUT] = {}
    for name, module in list(targets.items()):
        folded_weight = None
        folded_bias = None
        if fold_bn:
            bn = lookup_paired_bn(model, name)
            if bn is not None:
                folded_weight, folded_bias = fold_bn_parameters(module, bn)
        wrapper = NativeScalarLevelLUT(
            name=name,
            module=module,
            input_range=ranges[name],
            integer_max=integer_max,
            uniform_bits=uniform_bits,
            input_chunk=input_chunk,
            folded_weight=folded_weight,
            folded_bias=folded_bias,
        )
        set_submodule(model, name, wrapper)
        if fold_bn:
            bn_name = paired_bn_name(name)
            if bn_name is not None and lookup_paired_bn(model, name) is not None:
                set_submodule(model, bn_name, torch.nn.Identity())
        wrappers[name] = wrapper
    return wrappers


def reset_wrapper_metrics(wrappers: Dict[str, NativeScalarLevelLUT]) -> None:
    for wrapper in wrappers.values():
        wrapper.executions = 0
        wrapper.clip_count = 0
        wrapper.input_count = 0
        wrapper.input_quant_sse = 0.0


def wrapper_summary(wrappers: Dict[str, NativeScalarLevelLUT]) -> Dict[str, object]:
    per_target = {name: wrapper.summary() for name, wrapper in wrappers.items()}
    table_entries = sum(wrapper.table_entries() for wrapper in wrappers.values())
    clip_count = sum(wrapper.clip_count for wrapper in wrappers.values())
    input_count = sum(wrapper.input_count for wrapper in wrappers.values())
    quant_sse = sum(wrapper.input_quant_sse for wrapper in wrappers.values())
    return {
        "table_scalar_entries": table_entries,
        "table_fp32_kib": table_entries * 4.0 / 1024.0,
        "targets_requested": len(wrappers),
        "targets_executed": sum(1 for wrapper in wrappers.values() if wrapper.executions > 0),
        "total_native_executions": sum(wrapper.executions for wrapper in wrappers.values()),
        "clip_rate": clip_count / input_count if input_count else 0.0,
        "input_quant_mse": quant_sse / input_count if input_count else 0.0,
        "per_target": per_target,
    }


@torch.no_grad()
def evaluate(clean_model, native_model, loader, device, wrappers, max_batches: Optional[int]) -> Dict[str, object]:
    clean_model.eval()
    native_model.eval()
    clean_top1 = ScalarMeter()
    native_top1 = ScalarMeter()
    kl = ScalarMeter()
    logit_mse = ScalarMeter()
    reset_wrapper_metrics(wrappers)
    for batch_index, (images, target) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        reset_snn(clean_model)
        clean = clean_model(images)
        reset_snn(native_model)
        native = native_model(images)
        count = int(target.numel())
        clean_top1.update(accuracy(clean, target, topk=(1,))[0], count)
        native_top1.update(accuracy(native, target, topk=(1,))[0], count)
        kl.update(
            float(
                F.kl_div(
                    F.log_softmax(native.float(), dim=1),
                    F.softmax(clean.float(), dim=1),
                    reduction="batchmean",
                ).item()
            ),
            count,
        )
        logit_mse.update(float(F.mse_loss(native.float(), clean.float()).item()), count)
    reset_snn(clean_model)
    reset_snn(native_model)
    local = wrapper_summary(wrappers)
    return {
        "clean_top1": clean_top1.mean,
        "native_top1": native_top1.mean,
        "drop_top1": clean_top1.mean - native_top1.mean,
        "kl_clean_to_native": kl.mean,
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
    clean_model = build_cifar10_model(root, model_cfg).to(device)
    native_model = build_cifar10_model(root, model_cfg).to(device)
    clean_checkpoint = load_checkpoint_if_available(clean_model, args.checkpoint)
    native_checkpoint = load_checkpoint_if_available(native_model, args.checkpoint)
    for model in (clean_model, native_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    clean_targets = discover_targets(clean_model, args.category)
    native_targets = discover_targets(native_model, args.category)
    native_lif_targets = discover_lif_targets(native_model)
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

    calibration_batches, stats = collect_input_ranges(clean_model, train_loader, device, clean_targets, args.calib_batches)
    ranges = {
        name: (float(stats[name].summary()["minimum"]), float(stats[name].summary()["maximum"]))
        for name in clean_targets
    }
    wrappers = replace_targets(
        native_model,
        native_targets,
        ranges,
        integer_max=args.integer_max,
        uniform_bits=args.uniform_bits,
        input_chunk=args.input_chunk,
        fold_bn=args.fold_bn,
    )
    lif_calibration_batches = 0
    lif_ranges: Dict[str, Dict[str, object]] = {}
    if args.replace_lif and args.lif_lut_bits > 0:
        lif_calibration_batches, lif_x_stats, lif_v_stats = collect_lif_ranges(
            native_model,
            train_loader,
            device,
            native_lif_targets,
            args.calib_batches,
        )
        for name in native_lif_targets:
            x_summary = lif_x_stats[name].summary()
            x_integer = None
            if float(x_summary["integer_max_error"]) <= args.lif_integer_tolerance:
                x_integer = (
                    math.floor(float(x_summary["minimum"])) - args.lif_integer_margin,
                    math.ceil(float(x_summary["maximum"])) + args.lif_integer_margin,
                )
            lif_ranges[name] = {
                "x": expanded_range(lif_x_stats[name], args.lif_range_margin),
                "v": expanded_range(lif_v_stats[name], args.lif_range_margin),
                "x_integer": x_integer,
                "x_integer_max_error": float(x_summary["integer_max_error"]),
            }
    lif_replacements = (
        replace_lif_targets(
            native_model,
            native_lif_targets,
            lif_lut_bits=args.lif_lut_bits,
            lif_ranges=lif_ranges,
        )
        if args.replace_lif
        else {}
    )
    remaining_original_targets, original_counter, original_handles = attach_remaining_original_counter(
        native_model, native_targets.keys()
    )
    folded_bn_names = [
        name
        for name in (paired_bn_name(target_name) for target_name in native_targets)
        if name is not None
    ]
    remaining_folded_bn_targets, bn_counter, bn_handles = attach_remaining_bn_counter(native_model, folded_bn_names)
    remaining_lif_targets, lif_counter, lif_handles = attach_remaining_lif_counter(native_model, native_lif_targets.keys())
    try:
        metrics = evaluate(clean_model, native_model, validation_loader, device, wrappers, args.max_eval_batches)
    finally:
        for handle in original_handles + bn_handles + lif_handles:
            handle.remove()
    original_target_eval_calls = int(original_counter["calls"])
    folded_bn_eval_calls = int(bn_counter["calls"])
    original_lif_eval_calls = int(lif_counter["calls"])
    native_lif_targets_executed = sum(
        1 for module in lif_replacements.values() if module.executions > 0
    )
    native_lif_executions = sum(module.executions for module in lif_replacements.values())
    lif_lut_modules = [
        module for module in lif_replacements.values() if isinstance(module, QuantizedTransitionLUT)
    ]
    lif_lut_clip_count = sum(module.clip_count for module in lif_lut_modules)
    lif_lut_input_count = sum(module.input_count for module in lif_lut_modules)
    lif_lut_quant_sse = sum(module.quant_sse for module in lif_lut_modules)
    lif_lut_table_entries = sum(module.table_entries() for module in lif_lut_modules)
    lif_lut_clip_rate = lif_lut_clip_count / lif_lut_input_count if lif_lut_input_count else 0.0
    lif_lut_quant_mse = lif_lut_quant_sse / lif_lut_input_count if lif_lut_input_count else 0.0
    gate = {
        "clean_baseline_pass": abs(metrics["clean_top1"] - args.expected_clean_top1) <= args.clean_tolerance,
        "drop_top1_pass": metrics["drop_top1"] <= args.max_drop,
        "clip_rate_pass": metrics["clip_rate"] <= args.max_clip_rate,
        "all_native_targets_executed": metrics["targets_executed"] == metrics["targets_requested"],
        "original_target_modules_removed": remaining_original_targets == 0,
        "original_target_eval_calls_zero": original_target_eval_calls == 0,
        "folded_bn_modules_removed": (not args.fold_bn) or remaining_folded_bn_targets == 0,
        "folded_bn_eval_calls_zero": (not args.fold_bn) or folded_bn_eval_calls == 0,
        "original_lif_modules_removed": (not args.replace_lif) or remaining_lif_targets == 0,
        "original_lif_eval_calls_zero": (not args.replace_lif) or original_lif_eval_calls == 0,
        "all_native_lif_targets_executed": (not args.replace_lif) or (
            native_lif_targets_executed == len(lif_replacements)
        ),
        "lif_lut_clip_rate_pass": args.lif_lut_bits <= 0
        or lif_lut_clip_rate <= args.max_lif_clip_rate,
        "finite": all(
            math.isfinite(float(metrics[key]))
            for key in ("clean_top1", "native_top1", "clip_rate", "logit_mse")
        ),
    }
    gate["pass"] = all(bool(value) for value in gate.values())
    payload = {
        "experiment": "qkformer_native_affine_lut_probe",
        "claim": (
            "LUT-native replacement for frozen learned affine operators. Target "
            "Conv/Linear modules are replaced by lookup-and-accumulate modules "
            "during native evaluation. Optional BN folding removes paired inference "
            "BatchNorm modules; optional LIF replacement uses either an exact native "
            "state transition or a finite quantized (v, x) transition table. Pooling, "
            "residual, and attention matrix products remain unchanged."
        ),
        "checkpoint": {"clean": clean_checkpoint, "native": native_checkpoint},
        "category": args.category,
        "targets": list(native_targets),
        "calibration_batches": calibration_batches,
        "args": vars(args),
        "input_ranges": ranges,
        "remaining_original_targets": remaining_original_targets,
        "original_target_eval_calls": original_target_eval_calls,
        "fold_bn": bool(args.fold_bn),
        "folded_bn_targets": folded_bn_names if args.fold_bn else [],
        "remaining_folded_bn_targets": remaining_folded_bn_targets if args.fold_bn else None,
        "folded_bn_eval_calls": folded_bn_eval_calls if args.fold_bn else None,
        "replace_lif": bool(args.replace_lif),
        "lif_mode": (
            f"quantized_transition_{args.lif_lut_bits}bit"
            if args.replace_lif and args.lif_lut_bits > 0
            else "native_exact"
            if args.replace_lif
            else "original"
        ),
        "lif_calibration_batches": lif_calibration_batches,
        "lif_ranges": lif_ranges,
        "lif_targets": list(native_lif_targets) if args.replace_lif else [],
        "remaining_lif_targets": remaining_lif_targets if args.replace_lif else None,
        "original_lif_eval_calls": original_lif_eval_calls if args.replace_lif else None,
        "native_lif_targets_executed": native_lif_targets_executed if args.replace_lif else None,
        "native_lif_executions": native_lif_executions if args.replace_lif else None,
        "lif_lut_clip_rate": lif_lut_clip_rate if args.lif_lut_bits > 0 else None,
        "lif_lut_quant_mse": lif_lut_quant_mse if args.lif_lut_bits > 0 else None,
        "lif_lut_table_scalar_entries": lif_lut_table_entries if args.lif_lut_bits > 0 else None,
        "lif_lut_fp32_kib": lif_lut_table_entries * 4.0 / 1024.0 if args.lif_lut_bits > 0 else None,
        "lif_lut_per_target": (
            {name: module.summary() for name, module in lif_replacements.items()}
            if args.lif_lut_bits > 0
            else {}
        ),
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
    parser.add_argument("--fold-bn", action="store_true")
    parser.add_argument("--replace-lif", action="store_true")
    parser.add_argument("--lif-lut-bits", type=int, default=0)
    parser.add_argument("--lif-range-margin", type=float, default=0.05)
    parser.add_argument("--lif-integer-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--lif-integer-margin", type=int, default=1)
    parser.add_argument("--max-lif-clip-rate", type=float, default=0.0001)
    parser.add_argument("--max-drop", type=float, default=1.50)
    parser.add_argument("--max-clip-rate", type=float, default=0.0001)
    parser.add_argument("--expected-clean-top1", type=float, default=77.78)
    parser.add_argument("--clean-tolerance", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-cuda-index", type=int, default=0)
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "result_dir": str(Path(args.result_dir).resolve()),
                "category": args.category,
                "time_step": args.time_step,
                "gate": payload["gate"],
                "drop_top1": payload["metrics"]["drop_top1"],
                "clip_rate": payload["metrics"]["clip_rate"],
                "remaining_original_targets": payload["remaining_original_targets"],
                "original_target_eval_calls": payload["original_target_eval_calls"],
                "fold_bn": payload["fold_bn"],
                "remaining_folded_bn_targets": payload["remaining_folded_bn_targets"],
                "folded_bn_eval_calls": payload["folded_bn_eval_calls"],
                "replace_lif": payload["replace_lif"],
                "lif_mode": payload["lif_mode"],
                "remaining_lif_targets": payload["remaining_lif_targets"],
                "original_lif_eval_calls": payload["original_lif_eval_calls"],
                "native_lif_targets_executed": payload["native_lif_targets_executed"],
                "native_lif_executions": payload["native_lif_executions"],
                "lif_lut_clip_rate": payload["lif_lut_clip_rate"],
                "lif_lut_quant_mse": payload["lif_lut_quant_mse"],
                "lif_lut_table_scalar_entries": payload["lif_lut_table_scalar_entries"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
