#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from qkformer_lut.cross_arch_hooks import DensePrototypeLUT
from tools.qkformer_lut_e2_replace import (
    accuracy,
    build_cifar10_model,
    build_loader,
    _as_binary,
    _bin_index,
    load_checkpoint_if_available,
    spiking_self_full_address,
    token_qk_full_address,
)


def reset_snn(model: torch.nn.Module) -> None:
    try:
        from spikingjelly.clock_driven import functional
    except Exception:
        return
    functional.reset_net(model)


class ScalarMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int) -> None:
        self.total += float(value) * int(count)
        self.count += int(count)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


class TensorMoment:
    def __init__(self) -> None:
        self.sum: Optional[torch.Tensor] = None
        self.sumsq: Optional[torch.Tensor] = None
        self.count = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor, time_steps: int, batch_size: int) -> None:
        tb, c, n = x.shape
        value = x.detach().reshape(time_steps, batch_size, c, n).to(torch.float64)
        current_sum = value.sum(dim=(1, 3)).cpu()
        current_sumsq = (value * value).sum(dim=(1, 3)).cpu()
        current_count = int(batch_size * n)
        if self.sum is None:
            self.sum = current_sum
            self.sumsq = current_sumsq
        else:
            self.sum += current_sum
            self.sumsq += current_sumsq
        self.count += current_count

    def mean_std(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.sum is None or self.sumsq is None or self.count <= 0:
            raise RuntimeError("empty moment")
        mean = self.sum / float(self.count)
        var = torch.clamp(self.sumsq / float(self.count) - mean * mean, min=1.0e-12)
        return mean.to(device=device, dtype=dtype), var.sqrt().to(device=device, dtype=dtype)


class QKFormerCurrentLUTHook:
    def __init__(
        self,
        model: torch.nn.Module,
        targets: List[str],
        mode: str = "aligned_lut",
        token_bins: int = 8,
        channel_bins: int = 8,
        population_bins: int = 4,
        ssa_address_mode: str = "qk",
        ssa_context_groups: int = 6,
        ssa_context_bins: int = 4,
        projection_group_size: int = 8,
        min_support: int = 2,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.targets = set(targets)
        self.mode = mode
        self.token_bins = int(token_bins)
        self.channel_bins = int(channel_bins)
        self.population_bins = int(population_bins)
        self.ssa_address_mode = str(ssa_address_mode)
        self.ssa_context_groups = int(ssa_context_groups)
        self.ssa_context_bins = int(ssa_context_bins)
        self.projection_group_size = int(projection_group_size)
        self.min_support = int(min_support)
        self.seed = int(seed)
        self.enabled = False
        self.collect_calibration = False
        self.collect_lut_moments = False
        self.alpha = 1.0
        self.calibration_mode = "none"
        self.buffers: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.luts: Dict[str, DensePrototypeLUT] = {}
        self.orig_moments: Dict[str, TensorMoment] = defaultdict(TensorMoment)
        self.lut_moments: Dict[str, TensorMoment] = defaultdict(TensorMoment)
        self.device_cache: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}
        self.projection_lut_cache: Dict[Tuple[str, str, str], List[torch.Tensor]] = {}
        self.metrics = self._new_metrics()
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _new_metrics(self):
        return {
            "current_mse": ScalarMeter(),
            "current_cosine": ScalarMeter(),
            "hit": 0,
            "total": 0,
            "spike_zero": 0,
            "spike_total": 0,
        }

    def reset_metrics(self) -> None:
        self.metrics = self._new_metrics()

    def finalize(self) -> None:
        for lut in self.luts.values():
            lut.finalize()
        self.device_cache.clear()

    def supported_scalar_entries(self) -> int:
        if self.calibration_mode == "group_lut":
            return self.projection_group_scalar_entries()
        return sum(lut.supported_entries for lut in self.luts.values())

    def projection_group_scalar_entries(self) -> int:
        total = 0
        for prefix in self.targets:
            conv = getattr(self.model.get_submodule(prefix), "proj_conv")
            for start in range(0, int(conv.in_channels), self.projection_group_size):
                width = min(self.projection_group_size, int(conv.in_channels) - start)
                total += (1 << width) * int(conv.out_channels)
        return total

    def memory_kib(self) -> float:
        return self.supported_scalar_entries() * 4.0 / 1024.0

    def set_passthrough(self) -> None:
        self.enabled = False
        self.collect_calibration = False
        self.collect_lut_moments = False
        self.alpha = 0.0
        self.calibration_mode = "none"

    def set_eval(self, alpha: float, calibration_mode: str = "none") -> None:
        if calibration_mode not in {"none", "moment", "temporal", "channel", "tcslu", "group_lut"}:
            raise ValueError(f"unsupported calibration_mode={calibration_mode}")
        self.enabled = True
        self.collect_calibration = False
        self.collect_lut_moments = False
        self.alpha = float(alpha)
        self.calibration_mode = calibration_mode
        self.reset_metrics()

    def _register(self) -> None:
        for name, module in self.model.named_modules():
            if name not in self.targets:
                continue
            if module.__class__.__name__ not in {"Token_QK_Attention", "Spiking_Self_Attention"}:
                continue
            for child_name in ("q_lif", "k_lif", "attn_lif", "proj_conv"):
                child = getattr(module, child_name, None)
                if child is None:
                    continue
                kind = child_name.replace("_lif", "").replace("_conv", "")
                self.handles.append(child.register_forward_hook(self._make_hook(name, kind)))

    def _make_hook(self, prefix: str, kind: str):
        def hook(_module, _inputs, output):
            self.buffers[prefix][kind] = output.detach()
            if kind in {"q", "k", "attn"} and (self.enabled or self.collect_calibration):
                self.metrics["spike_zero"] += int((output.detach() <= 0).sum().item())
                self.metrics["spike_total"] += int(output.numel())
            if kind == "proj":
                try:
                    return self._consume(prefix, output)
                finally:
                    if not self.collect_lut_moments:
                        self.buffers[prefix].clear()
            return None

        return hook

    def _address(self, prefix: str, output: torch.Tensor):
        buf = self.buffers[prefix]
        module = self.model.get_submodule(prefix)
        cls = module.__class__.__name__
        if cls == "Token_QK_Attention":
            if "q" not in buf or "k" not in buf or "attn" not in buf:
                raise RuntimeError(f"missing q/k/attn for {prefix}")
            address, response, layout, kind = token_qk_full_address(
                buf["q"], buf["k"], buf["attn"], output, self.token_bins, self.channel_bins
            )
            heads = int(getattr(module, "num_heads", 1))
            address_space = heads * self.token_bins * self.channel_bins * 4
            coarse_address, coarse_address_space = self._token_qk_coarse_address(buf["q"], buf["attn"])
        elif cls == "Spiking_Self_Attention":
            if "q" not in buf or "k" not in buf:
                raise RuntimeError(f"missing q/k for {prefix}")
            if self.ssa_address_mode == "attn_context":
                if "attn" not in buf:
                    raise RuntimeError(f"missing attn for {prefix}")
                address, response, layout, kind, address_space = self._spiking_self_context_address(
                    module,
                    buf["attn"],
                    output,
                )
            else:
                address, response, layout, kind = spiking_self_full_address(
                    module, buf["q"], buf["k"], output, self.token_bins, self.channel_bins, self.population_bins
                )
                heads = int(getattr(module, "num_heads", 1))
                address_space = heads * self.token_bins * self.channel_bins * 4 * self.population_bins * self.population_bins
            coarse_address, coarse_address_space = self._spiking_self_coarse_address(module, buf["q"])
        else:
            raise ValueError(cls)
        if self.mode == "token_channel_mean":
            return coarse_address, response, layout, kind, coarse_address_space
        return address, response, layout, kind, address_space

    def _spiking_self_context_address(
        self,
        module: torch.nn.Module,
        attn_output: torch.Tensor,
        output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int, int], str, int]:
        attn = _as_binary(attn_output)
        t, b, c, n = attn.shape
        groups = self.ssa_context_groups
        bins = self.ssa_context_bins
        if groups <= 0 or bins <= 1:
            raise ValueError("SSA context groups must be positive and bins must exceed one")
        if c % groups != 0:
            raise ValueError(f"SSA channels {c} must be divisible by context groups {groups}")
        group_depth = c // groups
        grouped = attn.reshape(t, b, groups, group_depth, n).sum(dim=3)
        group_bin = _bin_index(grouped, bins, group_depth + 1)
        context = torch.zeros((t, b, n), device=attn.device, dtype=torch.long)
        for group_index in range(groups):
            context = context * bins + group_bin[:, :, group_index, :]
        heads = int(getattr(module, "num_heads", 1))
        depth = c // heads
        output_channel = torch.arange(c, device=attn.device).reshape(1, 1, c, 1).expand(t, b, c, n)
        address = context.unsqueeze(2) * c + output_channel
        proj = output.reshape(t, b, c, n).float()
        response = proj.reshape(t, b, heads, depth, n).permute(0, 1, 2, 4, 3)
        address = address.reshape(t, b, heads, depth, n).permute(0, 1, 2, 4, 3)
        address_space = (bins ** groups) * c
        return address.reshape(-1), response.reshape(-1), (t, b, c, n), "spiking_self", address_space

    def _token_qk_coarse_address(self, q_output: torch.Tensor, attn_output: torch.Tensor) -> Tuple[torch.Tensor, int]:
        q_raw = _as_binary(q_output)
        gate_raw = _as_binary(attn_output)
        t, b, c, n = q_raw.shape
        heads = gate_raw.shape[2]
        depth = c // heads
        token_idx = torch.arange(n, device=q_raw.device).reshape(1, 1, 1, 1, n).expand(t, b, heads, depth, n)
        channel_idx = torch.arange(depth, device=q_raw.device).reshape(1, 1, 1, depth, 1).expand(t, b, heads, depth, n)
        head_idx = torch.arange(heads, device=q_raw.device).reshape(1, 1, heads, 1, 1).expand(t, b, heads, depth, n)
        token_bin = _bin_index(token_idx, self.token_bins, n)
        channel_bin = _bin_index(channel_idx, self.channel_bins, depth)
        coarse = (head_idx * self.token_bins + token_bin) * self.channel_bins + channel_bin
        return coarse.reshape(-1), heads * self.token_bins * self.channel_bins

    def _spiking_self_coarse_address(self, module: torch.nn.Module, q_output: torch.Tensor) -> Tuple[torch.Tensor, int]:
        q_raw = _as_binary(q_output)
        t, b, c, n = q_raw.shape
        heads = int(getattr(module, "num_heads", 1))
        depth = c // heads
        token_idx = torch.arange(n, device=q_raw.device).reshape(1, 1, 1, n, 1).expand(t, b, heads, n, depth)
        channel_idx = torch.arange(depth, device=q_raw.device).reshape(1, 1, 1, 1, depth).expand(t, b, heads, n, depth)
        head_idx = torch.arange(heads, device=q_raw.device).reshape(1, 1, heads, 1, 1).expand(t, b, heads, n, depth)
        token_bin = _bin_index(token_idx, self.token_bins, n)
        channel_bin = _bin_index(channel_idx, self.channel_bins, depth)
        coarse = (head_idx * self.token_bins + token_bin) * self.channel_bins + channel_bin
        return coarse.reshape(-1), heads * self.token_bins * self.channel_bins

    def _restore(self, pred: torch.Tensor, layout: Tuple[int, int, int, int], output: torch.Tensor, kind: str, prefix: str):
        t, b, c, n = layout
        if kind == "token_qk":
            return pred.reshape(t * b, c, n).reshape_as(output)
        heads = int(getattr(self.model.get_submodule(prefix), "num_heads", 1))
        depth = c // heads
        return pred.reshape(t, b, heads, n, depth).permute(0, 1, 2, 4, 3).reshape(t * b, c, n).reshape_as(output)

    def _consume(self, prefix: str, output: torch.Tensor):
        address, response, layout, kind, address_space = self._address(prefix, output)
        lut = self.luts.get(prefix)
        if lut is None:
            lut_mode = self.mode if self.mode in {"aligned_lut", "shuffled_address", "token_channel_mean"} else "aligned_lut"
            lut = DensePrototypeLUT(address_space, mode=lut_mode, min_support=self.min_support, seed=self.seed)
            self.luts[prefix] = lut
        if self.collect_calibration:
            lut.update(address, response)
            self.orig_moments[prefix].update(output, layout[0], layout[1])
            return None
        if self.collect_lut_moments:
            pred, _seen = self._predict(prefix, address)
            pred_output = self._restore(pred, layout, output, kind, prefix)
            self.lut_moments[prefix].update(pred_output, layout[0], layout[1])
            return None
        if not self.enabled:
            return None
        if self.calibration_mode == "group_lut":
            replacement = self._grouped_projection(prefix, output).to(output.dtype)
            seen = torch.ones(response.numel(), device=output.device, dtype=torch.bool)
            blended = output + self.alpha * (replacement - output)
            self._update_metrics(output, replacement, seen)
            return blended
        pred, seen = self._predict(prefix, address)
        replacement = self._restore(pred, layout, output, kind, prefix).to(output.dtype)
        replacement = self._calibrate_replacement(prefix, replacement, layout)
        blended = output + self.alpha * (replacement - output)
        self._update_metrics(output, replacement, seen)
        return blended

    def _grouped_projection(self, prefix: str, output: torch.Tensor) -> torch.Tensor:
        module = self.model.get_submodule(prefix)
        cls = module.__class__.__name__
        buf = self.buffers[prefix]
        if cls == "Token_QK_Attention":
            k = _as_binary(buf["k"])
            gate = _as_binary(buf["attn"])
            t, b, c, n = k.shape
            heads = gate.shape[2]
            depth = c // heads
            projection_input = (
                gate.reshape(t, b, heads, 1, n)
                * k.reshape(t, b, heads, depth, n)
            ).reshape(t * b, c, n)
        elif cls == "Spiking_Self_Attention":
            projection_input = _as_binary(buf["attn"])
            t, b, c, n = projection_input.shape
            projection_input = projection_input.reshape(t * b, c, n)
        else:
            raise ValueError(cls)

        conv = getattr(module, "proj_conv")
        weight = conv.weight.detach().reshape(conv.out_channels, conv.in_channels)
        cache_key = (prefix, str(output.device), str(output.dtype))
        tables = self.projection_lut_cache.get(cache_key)
        if tables is None:
            tables = []
            for start in range(0, int(conv.in_channels), self.projection_group_size):
                width = min(self.projection_group_size, int(conv.in_channels) - start)
                patterns = torch.arange(1 << width, device=output.device, dtype=torch.long)
                shifts = torch.arange(width, device=output.device, dtype=torch.long)
                bits = ((patterns[:, None] >> shifts[None, :]) & 1).to(output.dtype)
                group_weight = weight[:, start : start + width].to(device=output.device, dtype=output.dtype)
                tables.append(bits @ group_weight.transpose(0, 1))
            self.projection_lut_cache[cache_key] = tables

        x = projection_input.to(device=output.device, dtype=torch.long).permute(0, 2, 1)
        if conv.bias is None:
            result = torch.zeros(
                (x.shape[0], x.shape[1], int(conv.out_channels)),
                device=output.device,
                dtype=output.dtype,
            )
        else:
            result = conv.bias.detach().to(device=output.device, dtype=output.dtype).reshape(1, 1, -1)
            result = result.expand(x.shape[0], x.shape[1], -1).clone()
        for group_index, start in enumerate(range(0, int(conv.in_channels), self.projection_group_size)):
            width = min(self.projection_group_size, int(conv.in_channels) - start)
            powers = 1 << torch.arange(width, device=output.device, dtype=torch.long)
            pattern = (x[:, :, start : start + width] * powers).sum(dim=-1)
            result = result + tables[group_index][pattern]
        return result.permute(0, 2, 1).reshape_as(output)

    def _predict(self, prefix: str, address: torch.Tensor):
        lut = self.luts[prefix]
        if lut.mean is None or lut.support is None:
            raise RuntimeError("finalize before predict")
        key = (prefix, str(address.device))
        cache = self.device_cache.get(key)
        if cache is None:
            cache = {
                "mean": lut.mean.to(address.device),
                "support": lut.support.to(address.device),
                "global": torch.tensor(float(lut.global_mean.item()), device=address.device, dtype=torch.float32),
            }
            if lut.mode == "shuffled_address":
                gen = torch.Generator(device="cpu").manual_seed(self.seed)
                cache["perm"] = torch.randperm(lut.address_space, generator=gen, dtype=torch.long).to(address.device)
            if self.mode == "random_table":
                gen = torch.Generator(device="cpu").manual_seed(self.seed)
                source = lut.mean.detach().cpu()
                support = lut.support.detach().cpu()
                values = source[support]
                if values.numel() == 0:
                    random_mean = source
                else:
                    draw = torch.randint(values.numel(), (lut.address_space,), generator=gen)
                    random_mean = values[draw].to(torch.float32)
                cache["random_mean"] = random_mean.to(address.device)
            self.device_cache[key] = cache
        query = address.long()
        if "perm" in cache:
            query = cache["perm"][query]
        if self.mode == "global_mean":
            seen = torch.ones_like(query, dtype=torch.bool)
            pred = torch.full_like(query, float(cache["global"].item()), dtype=torch.float32)
            return pred, seen
        seen = cache["support"][query]
        pred = cache["random_mean"][query] if self.mode == "random_table" else cache["mean"][query]
        pred = torch.where(seen, pred, torch.full_like(pred, float(cache["global"].item())))
        return pred, seen

    def _calibrate_replacement(self, prefix: str, x: torch.Tensor, layout: Tuple[int, int, int, int]) -> torch.Tensor:
        if self.calibration_mode == "none":
            return x
        if self.calibration_mode == "temporal":
            return self._temporal_match(prefix, x, layout)
        if self.calibration_mode == "channel":
            return self._channel_match(prefix, x, layout)
        if self.calibration_mode in {"moment", "tcslu"}:
            return self._moment_match(prefix, x, layout)
        return x

    def _moment_match(self, prefix: str, x: torch.Tensor, layout: Tuple[int, int, int, int]) -> torch.Tensor:
        if prefix not in self.orig_moments or prefix not in self.lut_moments:
            return x
        t, b, c, n = layout
        om, os = self.orig_moments[prefix].mean_std(x.device, x.dtype)
        lm, ls = self.lut_moments[prefix].mean_std(x.device, x.dtype)
        y = x.reshape(t, b, c, n)
        shape = (t, 1, c, 1)
        y = (y - lm.reshape(shape)) / (ls.reshape(shape) + 1.0e-6) * os.reshape(shape) + om.reshape(shape)
        return y.reshape_as(x)

    def _temporal_match(self, prefix: str, x: torch.Tensor, layout: Tuple[int, int, int, int]) -> torch.Tensor:
        if prefix not in self.orig_moments or prefix not in self.lut_moments:
            return x
        t, b, c, n = layout
        om, os = self.orig_moments[prefix].mean_std(x.device, x.dtype)
        lm, ls = self.lut_moments[prefix].mean_std(x.device, x.dtype)
        y = x.reshape(t, b, c, n)
        om_t = om.mean(dim=1).reshape(t, 1, 1, 1)
        lm_t = lm.mean(dim=1).reshape(t, 1, 1, 1)
        os_t = os.mean(dim=1).reshape(t, 1, 1, 1)
        ls_t = ls.mean(dim=1).reshape(t, 1, 1, 1)
        y = (y - lm_t) / (ls_t + 1.0e-6) * os_t + om_t
        return y.reshape_as(x)

    def _channel_match(self, prefix: str, x: torch.Tensor, layout: Tuple[int, int, int, int]) -> torch.Tensor:
        if prefix not in self.orig_moments or prefix not in self.lut_moments:
            return x
        t, b, c, n = layout
        om, os = self.orig_moments[prefix].mean_std(x.device, x.dtype)
        lm, ls = self.lut_moments[prefix].mean_std(x.device, x.dtype)
        y = x.reshape(t, b, c, n)
        om_c = om.mean(dim=0).reshape(1, 1, c, 1)
        lm_c = lm.mean(dim=0).reshape(1, 1, c, 1)
        os_c = os.mean(dim=0).reshape(1, 1, c, 1)
        ls_c = ls.mean(dim=0).reshape(1, 1, c, 1)
        y = (y - lm_c) / (ls_c + 1.0e-6) * os_c + om_c
        return y.reshape_as(x)

    def _update_metrics(self, output: torch.Tensor, replacement: torch.Tensor, seen: torch.Tensor) -> None:
        n = int(output.numel())
        of = output.detach().float().reshape(1, -1)
        rf = replacement.detach().float().reshape(1, -1)
        self.metrics["current_mse"].update(float(F.mse_loss(rf, of).item()), n)
        self.metrics["current_cosine"].update(float(F.cosine_similarity(rf, of, dim=1).item()), n)
        self.metrics["hit"] += int(seen.detach().bool().sum().item())
        self.metrics["total"] += int(seen.numel())

    def summary(self) -> Dict[str, float]:
        total = self.metrics["total"]
        return {
            "current_mse": self.metrics["current_mse"].mean,
            "current_cosine": self.metrics["current_cosine"].mean,
            "hit_rate": self.metrics["hit"] / total if total else 0.0,
            "fallback_rate": 1.0 - (self.metrics["hit"] / total) if total else 1.0,
            "spike_sparsity": self.metrics["spike_zero"] / self.metrics["spike_total"] if self.metrics["spike_total"] else 0.0,
            "supported_scalar_entries": float(self.supported_scalar_entries()),
            "memory_kib": self.memory_kib(),
        }


class QKFormerCurrentLUTGroup:
    def __init__(self, hooks: List[QKFormerCurrentLUTHook]) -> None:
        self.hooks = hooks

    def close(self) -> None:
        for hook in self.hooks:
            hook.close()

    def reset_metrics(self) -> None:
        for hook in self.hooks:
            hook.reset_metrics()

    def set_passthrough(self) -> None:
        for hook in self.hooks:
            hook.set_passthrough()

    def set_eval(self, alpha: float, calibration_mode: str = "none") -> None:
        for hook in self.hooks:
            hook.set_eval(alpha=alpha, calibration_mode=calibration_mode)

    def summary(self) -> Dict[str, object]:
        per_target = {
            next(iter(hook.targets)): hook.summary()
            for hook in self.hooks
        }
        current_count = sum(hook.metrics["current_mse"].count for hook in self.hooks)
        cosine_count = sum(hook.metrics["current_cosine"].count for hook in self.hooks)
        hit = sum(int(hook.metrics["hit"]) for hook in self.hooks)
        total = sum(int(hook.metrics["total"]) for hook in self.hooks)
        spike_zero = sum(int(hook.metrics["spike_zero"]) for hook in self.hooks)
        spike_total = sum(int(hook.metrics["spike_total"]) for hook in self.hooks)
        current_total = sum(hook.metrics["current_mse"].total for hook in self.hooks)
        cosine_total = sum(hook.metrics["current_cosine"].total for hook in self.hooks)
        supported_entries = sum(hook.supported_scalar_entries() for hook in self.hooks)
        return {
            "current_mse": current_total / current_count if current_count else 0.0,
            "current_cosine": cosine_total / cosine_count if cosine_count else 0.0,
            "hit_rate": hit / total if total else 0.0,
            "fallback_rate": 1.0 - (hit / total) if total else 1.0,
            "spike_sparsity": spike_zero / spike_total if spike_total else 0.0,
            "supported_scalar_entries": float(supported_entries),
            "memory_kib": supported_entries * 4.0 / 1024.0,
            "per_target_metrics": per_target,
        }


@torch.no_grad()
def calibrate(model, loader, device, hook: QKFormerCurrentLUTHook, batches: int) -> int:
    model.eval()
    hook.collect_calibration = True
    hook.enabled = False
    seen = 0
    for images, _target in loader:
        if seen >= batches:
            break
        reset_snn(model)
        model(images.to(device, non_blocking=True).float())
        seen += 1
    reset_snn(model)
    hook.collect_calibration = False
    hook.finalize()
    return seen


@torch.no_grad()
def collect_lut_moments(model, loader, device, hook: QKFormerCurrentLUTHook, batches: int) -> int:
    model.eval()
    hook.collect_lut_moments = True
    hook.enabled = False
    seen = 0
    for images, _target in loader:
        if seen >= batches:
            break
        reset_snn(model)
        model(images.to(device, non_blocking=True).float())
        seen += 1
    reset_snn(model)
    hook.collect_lut_moments = False
    return seen


@torch.no_grad()
def calibrate_sequential(
    model,
    loader,
    device,
    hooks: List[QKFormerCurrentLUTHook],
    batches: int,
) -> Tuple[List[int], List[int]]:
    calib_counts: List[int] = []
    moment_counts: List[int] = []
    for index, hook in enumerate(hooks):
        for previous in hooks[:index]:
            previous.set_eval(alpha=1.0, calibration_mode="moment")
        for following in hooks[index + 1 :]:
            following.set_passthrough()
        hook.set_passthrough()
        calib_counts.append(calibrate(model, loader, device, hook, batches))
        moment_counts.append(collect_lut_moments(model, loader, device, hook, batches))
        hook.set_eval(alpha=1.0, calibration_mode="moment")
    for hook in hooks:
        hook.set_passthrough()
    return calib_counts, moment_counts


def apply_input_corruption(
    images: torch.Tensor,
    corruption: str,
    severity: int,
    seed: int,
    batch_idx: int,
) -> torch.Tensor:
    if corruption == "none":
        return images
    severity = int(severity)
    if severity not in {1, 2}:
        raise ValueError(f"unsupported corruption severity: {severity}")
    if corruption == "noise":
        sigma = 0.08 if severity == 1 else 0.16
        generator = torch.Generator(device=images.device)
        generator.manual_seed(int(seed) + int(batch_idx))
        noise = torch.randn(images.shape, device=images.device, dtype=images.dtype, generator=generator)
        return images + sigma * noise
    if corruption == "brightness":
        delta = 0.12 if severity == 1 else 0.24
        # Inputs are normalized CIFAR tensors; shift by the channel-normalized
        # equivalent of a small RGB brightness offset.
        std = torch.tensor((0.2470, 0.2435, 0.2616), device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        return images + delta / std
    if corruption == "blur":
        kernel = 3 if severity == 1 else 5
        return F.avg_pool2d(images, kernel_size=kernel, stride=1, padding=kernel // 2)
    raise ValueError(f"unsupported corruption: {corruption}")


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    hook: QKFormerCurrentLUTHook,
    method: str,
    alpha: float,
    max_batches: Optional[int],
    calibration_mode: str = "none",
    corruption: str = "none",
    corruption_severity: int = 1,
    corruption_seed: int = 42,
) -> Dict[str, object]:
    model.eval()
    hook.reset_metrics()
    clean_top1 = ScalarMeter()
    inj_top1 = ScalarMeter()
    kl = ScalarMeter()
    logit_mse = ScalarMeter()
    for idx, (images, target) in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        images = images.to(device, non_blocking=True).float()
        images = apply_input_corruption(images, corruption, corruption_severity, corruption_seed, idx)
        target = target.to(device, non_blocking=True)
        hook.set_passthrough()
        reset_snn(model)
        clean = model(images)
        hook.set_eval(alpha=alpha, calibration_mode=calibration_mode)
        reset_snn(model)
        injected = model(images)
        n = int(target.numel())
        clean_top1.update(accuracy(clean, target, topk=(1,))[0], n)
        inj_top1.update(accuracy(injected, target, topk=(1,))[0], n)
        kl.update(float(F.kl_div(F.log_softmax(injected.float(), dim=1), F.softmax(clean.float(), dim=1), reduction="batchmean").item()), n)
        logit_mse.update(float(F.mse_loss(injected.float(), clean.float()).item()), n)
    reset_snn(model)
    local = hook.summary()
    return {
        "method": method,
        "alpha": alpha,
        "calibration_mode": calibration_mode,
        "corruption": corruption,
        "corruption_severity": 0 if corruption == "none" else int(corruption_severity),
        "moment_match": calibration_mode in {"moment", "tcslu"},
        "clean_top1": clean_top1.mean,
        "injected_top1": inj_top1.mean,
        "drop_top1": clean_top1.mean - inj_top1.mean,
        "kl_clean_to_injected": kl.mean,
        "logit_mse": logit_mse.mean,
        **local,
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    root = Path(args.root).resolve()
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.local_cuda_index}" if torch.cuda.is_available() else "cpu")
    is_dvs = args.family == "cifar10dvs"
    model_cfg = {
        "family": args.family,
        "img_size": 128 if is_dvs else 32,
        "patch_size": 16 if is_dvs else 4,
        "dim": args.dim,
        "num_heads": 16 if is_dvs else 8,
        "mlp_ratio": 1 if is_dvs else 4,
        "in_channels": 2 if is_dvs else 3,
        "num_classes": 100 if args.family == "cifar100" else 10,
        "layer": args.layer,
        "time_step": args.time_step,
    }
    model = build_cifar10_model(root, model_cfg).to(device)
    ckpt = load_checkpoint_if_available(model, args.checkpoint)
    for param in model.parameters():
        param.requires_grad_(False)
    data_cfg = {
        "mode": args.family,
        "data_dir": args.data_dir,
        "split": "train",
        "batch_size": args.batch_size,
        "workers": args.workers,
        "shuffle": True,
        "seed": args.seed,
    }
    val_cfg = dict(data_cfg)
    val_cfg.update({"split": "validation", "shuffle": False})
    train_loader = build_loader(data_cfg, device)
    val_loader = build_loader(val_cfg, device)
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    if args.group_lut_only:
        hooks = [
            QKFormerCurrentLUTHook(
                model,
                targets=[target],
                mode=args.lut_mode,
                token_bins=args.token_bins,
                channel_bins=args.channel_bins,
                population_bins=args.population_bins,
                ssa_address_mode=args.ssa_address_mode,
                ssa_context_groups=args.ssa_context_groups,
                ssa_context_bins=args.ssa_context_bins,
                projection_group_size=args.projection_group_size,
                min_support=args.min_support,
                seed=args.seed,
            )
            for target in targets
        ]
        hook = QKFormerCurrentLUTGroup(hooks)
        calib_batches = 0
        moment_batches = 0
    elif args.calibration_schedule == "sequential":
        hooks = [
            QKFormerCurrentLUTHook(
                model,
                targets=[target],
                mode=args.lut_mode,
                token_bins=args.token_bins,
                channel_bins=args.channel_bins,
                population_bins=args.population_bins,
                ssa_address_mode=args.ssa_address_mode,
                ssa_context_groups=args.ssa_context_groups,
                ssa_context_bins=args.ssa_context_bins,
                projection_group_size=args.projection_group_size,
                min_support=args.min_support,
                seed=args.seed,
            )
            for target in targets
        ]
        calib_counts, moment_counts = calibrate_sequential(
            model,
            train_loader,
            device,
            hooks,
            args.calib_batches,
        )
        hook = QKFormerCurrentLUTGroup(hooks)
        calib_batches: object = calib_counts
        moment_batches: object = moment_counts
    else:
        hook = QKFormerCurrentLUTHook(
            model,
            targets=targets,
            mode=args.lut_mode,
            token_bins=args.token_bins,
            channel_bins=args.channel_bins,
            population_bins=args.population_bins,
            ssa_address_mode=args.ssa_address_mode,
            ssa_context_groups=args.ssa_context_groups,
            ssa_context_bins=args.ssa_context_bins,
            projection_group_size=args.projection_group_size,
            min_support=args.min_support,
            seed=args.seed,
        )
        calib_batches = calibrate(model, train_loader, device, hook, args.calib_batches)
        moment_batches = collect_lut_moments(model, train_loader, device, hook, args.calib_batches)
    rows = []
    alpha_values = [] if args.group_lut_only else [float(x) for x in args.alphas.split(",") if x.strip()]
    for alpha in alpha_values:
        row = evaluate(
            model,
            val_loader,
            device,
            hook,
            f"qkformer_current_alpha_{alpha:g}",
            alpha,
            None if args.max_eval_batches <= 0 else args.max_eval_batches,
            corruption=args.corruption,
            corruption_severity=args.corruption_severity,
            corruption_seed=args.corruption_seed,
        )
        rows.append(row)
        print(json.dumps({"method": row["method"], "top1": row["injected_top1"], "drop": row["drop_top1"]}), flush=True)
    if args.group_lut_only:
        method_specs = [
            (f"qkformer_current_{args.lut_mode}_grouped_projection_lut_hard", "group_lut"),
        ]
    else:
        labels = {
            "none": "static",
            "moment": "moment_matched",
            "temporal": "temporal_matched",
            "channel": "channel_matched",
            "tcslu": "tcslu_moment_temporal_channel_backoff",
            "group_lut": "grouped_projection",
        }
        requested_modes = [
            item.strip() for item in args.calibration_modes.split(",") if item.strip()
        ]
        method_specs = [
            (f"qkformer_current_{args.lut_mode}_{labels[mode]}_lut_hard", mode)
            for mode in requested_modes
        ]
    for method, calibration_mode in method_specs:
        row = evaluate(
            model,
            val_loader,
            device,
            hook,
            method,
            1.0,
            None if args.max_eval_batches <= 0 else args.max_eval_batches,
            calibration_mode=calibration_mode,
            corruption=args.corruption,
            corruption_severity=args.corruption_severity,
            corruption_seed=args.corruption_seed,
        )
        rows.append(row)
        print(json.dumps({"method": row["method"], "top1": row["injected_top1"], "drop": row["drop_top1"]}), flush=True)
    hook.close()
    write_csv(result_dir / "summary.csv", rows)
    payload = {
        "experiment": "qkformer_current_unit_probe",
        "claim": "small diagnostic: replace QKFormer proj_conv current via hook, preserving proj_bn/proj_lif; does not remove conv compute",
        "checkpoint": ckpt,
        "targets": targets,
        "lut_mode": args.lut_mode,
        "calibration_schedule": args.calibration_schedule,
        "calib_batches": calib_batches,
        "moment_batches": moment_batches,
        "args": vars(args),
        "rows": rows,
    }
    (result_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--family",
        choices=("cifar10", "cifar100", "cifar10dvs"),
        default="cifar10",
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--time-step", type=int, default=1)
    parser.add_argument("--targets", default="stage1.0.tssa")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--token-bins", type=int, default=8)
    parser.add_argument("--channel-bins", type=int, default=8)
    parser.add_argument("--population-bins", type=int, default=4)
    parser.add_argument("--ssa-address-mode", choices=("qk", "attn_context"), default="qk")
    parser.add_argument("--ssa-context-groups", type=int, default=6)
    parser.add_argument("--ssa-context-bins", type=int, default=4)
    parser.add_argument("--projection-group-size", type=int, default=8)
    parser.add_argument("--group-lut-only", action="store_true")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-cuda-index", type=int, default=0)
    parser.add_argument("--alphas", default="0.0,0.25,1.0")
    parser.add_argument(
        "--calibration-modes",
        default="none,moment,temporal,channel,tcslu,group_lut",
        help="comma-separated hard-replacement calibration modes",
    )
    parser.add_argument(
        "--calibration-schedule",
        choices=("independent", "sequential"),
        default="independent",
    )
    parser.add_argument(
        "--lut-mode",
        choices=("aligned_lut", "shuffled_address", "token_channel_mean", "global_mean", "random_table"),
        default="aligned_lut",
    )
    parser.add_argument("--corruption", choices=("none", "noise", "blur", "brightness"), default="none")
    parser.add_argument("--corruption-severity", type=int, choices=(1, 2), default=1)
    parser.add_argument("--corruption-seed", type=int, default=12345)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"result_dir": str(Path(args.result_dir).resolve()), "rows": len(payload["rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
