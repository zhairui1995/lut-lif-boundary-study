from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from .stats import BucketStats, VarianceStats


ATTENTION_CLASS_NAMES = {
    "Token_QK_Attention",
    "Spiking_Self_Attention",
    "FF_LiDiff_Attention",
}


def _stage_from_name(name: str) -> str:
    match = re.search(r"(stage\d+)", name)
    return match.group(1) if match else "unknown"


def _block_from_name(name: str) -> str:
    match = re.search(r"stage\d+\.(\d+)", name)
    return match.group(1) if match else "unknown"


def _sample_indices(total: int, max_records: int, device: torch.device) -> torch.Tensor:
    if total <= max_records:
        return torch.arange(total, device=device)
    return torch.randperm(total, device=device)[:max_records]


def _bin_index(values: torch.Tensor, bins: int, max_value: int) -> torch.Tensor:
    if max_value <= 0:
        return torch.zeros_like(values, dtype=torch.long)
    return torch.clamp((values.long() * bins) // max_value, min=0, max=bins - 1)


def _population_bin(popcount: torch.Tensor, bins: int, depth: int) -> torch.Tensor:
    if depth <= 0:
        return torch.zeros_like(popcount, dtype=torch.long)
    return torch.clamp((popcount.long() * bins) // (depth + 1), min=0, max=bins - 1)


def _as_binary(x: torch.Tensor) -> torch.Tensor:
    return (x.detach() > 0).to(torch.long)


def _flatten_proj(
    output: torch.Tensor,
    time_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    if output.ndim == 5:
        t, b, c, h, w = output.shape
        return output.reshape(t, b, c, h * w)
    if output.ndim == 4:
        return output
    if output.ndim == 3 and time_steps is not None and batch_size is not None:
        tb, c, n = output.shape
        if tb != time_steps * batch_size:
            raise ValueError(
                "Projected output batch mismatch: "
                f"shape={tuple(output.shape)}, T={time_steps}, B={batch_size}"
            )
        return output.reshape(time_steps, batch_size, c, n)
    raise ValueError(f"Unsupported projected output shape: {tuple(output.shape)}")


@dataclass
class ModuleDiagnostic:
    name: str
    kind: str
    stage: str
    block: str
    address_space: int
    buckets: BucketStats
    response: VarianceStats = field(default_factory=VarianceStats)
    candidate: VarianceStats = field(default_factory=VarianceStats)
    background: VarianceStats = field(default_factory=VarianceStats)
    q_rate: VarianceStats = field(default_factory=VarianceStats)
    k_rate: VarianceStats = field(default_factory=VarianceStats)
    gate_rate: VarianceStats = field(default_factory=VarianceStats)
    batches: int = 0

    def summary(self) -> Dict[str, object]:
        bucket_summary = self.buckets.summary()
        return {
            "name": self.name,
            "kind": self.kind,
            "stage": self.stage,
            "block": self.block,
            "batches": int(self.batches),
            "address_coverage": bucket_summary["address_coverage"],
            "bucket_occupancy": bucket_summary["bucket_occupancy"],
            "singleton_fraction": bucket_summary["singleton_fraction"],
            "conditional_variance": bucket_summary["conditional_variance"],
            "response_variance": self.response.variance,
            "candidate_variance": self.candidate.variance,
            "background_variance": self.background.variance,
            "q_spike_rate": self.q_rate.mean,
            "k_spike_rate": self.k_rate.mean,
            "gate_spike_rate": self.gate_rate.mean,
            "num_samples": bucket_summary["num_samples"],
            "unique_addresses": bucket_summary["unique_addresses"],
            "address_space": bucket_summary["address_space"],
        }


class QKAddressDiagnostic:
    def __init__(
        self,
        model: torch.nn.Module,
        token_bins: int = 8,
        channel_bins: int = 8,
        population_bins: int = 4,
        max_records_per_module_per_batch: int = 65536,
        response_child: str = "proj_lif",
        record_callback: Optional[
            Callable[[ModuleDiagnostic, torch.Tensor, torch.Tensor, torch.Tensor], None]
        ] = None,
    ) -> None:
        self.model = model
        self.token_bins = int(token_bins)
        self.channel_bins = int(channel_bins)
        self.population_bins = int(population_bins)
        self.max_records = int(max_records_per_module_per_batch)
        self.response_child = str(response_child)
        if self.response_child not in {"proj_lif", "proj_conv", "proj_bn"}:
            raise ValueError(f"unsupported Q/K diagnostic response child: {self.response_child}")
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.buffers: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.module_kinds: Dict[str, str] = {}
        self.module_stats: Dict[str, ModuleDiagnostic] = {}
        self.record_callback = record_callback
        self._register_hooks()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _register_hooks(self) -> None:
        modules = dict(self.model.named_modules())
        for name, module in modules.items():
            if module.__class__.__name__ not in ATTENTION_CLASS_NAMES:
                continue
            class_name = module.__class__.__name__
            kind = (
                "Token_QK_Attention"
                if class_name == "FF_LiDiff_Attention"
                else class_name
            )
            self.module_kinds[name] = kind
            for child_name in ("q_lif", "k_lif", "v_lif", "attn_lif", self.response_child):
                child = getattr(module, child_name, None)
                if child is None:
                    continue
                hook_kind = (
                    "proj"
                    if child_name == self.response_child
                    else child_name.replace("_lif", "")
                )
                handle = child.register_forward_hook(
                    self._make_hook(name, hook_kind)
                )
                self.handles.append(handle)

    def _make_hook(self, prefix: str, kind: str):
        def hook(_module, _inputs, output):
            self.buffers[prefix][kind] = output.detach()
            if kind == "proj":
                self._consume(prefix)
                self.buffers[prefix].clear()

        return hook

    def _ensure_module_stats(
        self, prefix: str, kind: str, address_space: int
    ) -> ModuleDiagnostic:
        stats = self.module_stats.get(prefix)
        if stats is None:
            stats = ModuleDiagnostic(
                name=prefix,
                kind=kind,
                stage=_stage_from_name(prefix),
                block=_block_from_name(prefix),
                address_space=address_space,
                buckets=BucketStats(address_space=address_space),
            )
            self.module_stats[prefix] = stats
        return stats

    @torch.no_grad()
    def _consume(self, prefix: str) -> None:
        buf = self.buffers[prefix]
        if "q" not in buf or "k" not in buf or "proj" not in buf:
            return
        kind = self.module_kinds.get(prefix, "unknown")
        if kind == "Token_QK_Attention" and "attn" in buf:
            self._consume_token_qk(prefix, buf)
        elif kind == "Spiking_Self_Attention":
            self._consume_spiking_self(prefix, buf)

    def _consume_token_qk(self, prefix: str, buf: Dict[str, torch.Tensor]) -> None:
        q_raw = _as_binary(buf["q"])
        k_raw = _as_binary(buf["k"])
        gate_raw = _as_binary(buf["attn"])
        t, b, c, n = q_raw.shape
        proj = _flatten_proj(buf["proj"], t, b).float()
        heads = gate_raw.shape[2]
        depth = c // heads
        q_heads = q_raw.reshape(t, b, heads, depth, n)
        k_heads = k_raw.reshape(t, b, heads, depth, n)
        gate = gate_raw.reshape(t, b, heads, 1, n)

        total = t * b * heads * depth * n
        idx = _sample_indices(total, self.max_records, q_raw.device)
        tmp = idx
        token_idx = tmp % n
        tmp = tmp // n
        channel_idx = tmp % depth
        tmp = tmp // depth
        head_idx = tmp % heads
        tmp = tmp // heads
        batch_idx = tmp % b
        time_idx = tmp // b

        q_gate = gate[time_idx, batch_idx, head_idx, 0, token_idx]
        k_bit = k_heads[time_idx, batch_idx, head_idx, channel_idx, token_idx]
        channel_abs = head_idx * depth + channel_idx
        response = proj[time_idx, batch_idx, channel_abs, token_idx]

        token_bin = _bin_index(token_idx, self.token_bins, n)
        channel_bin = _bin_index(channel_idx, self.channel_bins, depth)
        address = (((head_idx * self.token_bins + token_bin) * self.channel_bins + channel_bin) * 2 + q_gate) * 2 + k_bit
        address_space = heads * self.token_bins * self.channel_bins * 4
        candidate_mask = (q_gate > 0) | (k_bit > 0)

        stats = self._ensure_module_stats(prefix, "token_qk", address_space)
        self._update_stats(stats, address, response, candidate_mask)
        stats.q_rate.update(float(q_heads.float().mean().item()))
        stats.k_rate.update(float(k_heads.float().mean().item()))
        stats.gate_rate.update(float(gate.float().mean().item()))
        stats.batches += 1

    def _consume_spiking_self(self, prefix: str, buf: Dict[str, torch.Tensor]) -> None:
        q_raw = _as_binary(buf["q"])
        k_raw = _as_binary(buf["k"])
        t, b, c, n = q_raw.shape
        proj = _flatten_proj(buf["proj"], t, b).float()
        kind_heads = getattr(self.model.get_submodule(prefix), "num_heads", None)
        heads = int(kind_heads) if kind_heads else 1
        depth = c // heads
        q_heads = q_raw.transpose(-1, -2).reshape(t, b, n, heads, depth).permute(0, 1, 3, 2, 4)
        k_heads = k_raw.transpose(-1, -2).reshape(t, b, n, heads, depth).permute(0, 1, 3, 2, 4)
        q_pop = q_heads.sum(dim=-1)
        k_pop = k_heads.sum(dim=-1)

        total = t * b * heads * n * depth
        idx = _sample_indices(total, self.max_records, q_raw.device)
        tmp = idx
        channel_idx = tmp % depth
        tmp = tmp // depth
        token_idx = tmp % n
        tmp = tmp // n
        head_idx = tmp % heads
        tmp = tmp // heads
        batch_idx = tmp % b
        time_idx = tmp // b

        q_bit = q_heads[time_idx, batch_idx, head_idx, token_idx, channel_idx]
        k_bit = k_heads[time_idx, batch_idx, head_idx, token_idx, channel_idx]
        q_pop_bin = _population_bin(q_pop[time_idx, batch_idx, head_idx, token_idx], self.population_bins, depth)
        k_pop_bin = _population_bin(k_pop[time_idx, batch_idx, head_idx, token_idx], self.population_bins, depth)
        channel_abs = head_idx * depth + channel_idx
        response = proj[time_idx, batch_idx, channel_abs, token_idx]

        token_bin = _bin_index(token_idx, self.token_bins, n)
        channel_bin = _bin_index(channel_idx, self.channel_bins, depth)
        address = head_idx
        address = address * self.token_bins + token_bin
        address = address * self.channel_bins + channel_bin
        address = address * 2 + q_bit
        address = address * 2 + k_bit
        address = address * self.population_bins + q_pop_bin
        address = address * self.population_bins + k_pop_bin
        address_space = heads * self.token_bins * self.channel_bins * 4 * self.population_bins * self.population_bins
        candidate_mask = (q_bit > 0) | (k_bit > 0) | (q_pop_bin > 0) | (k_pop_bin > 0)

        stats = self._ensure_module_stats(prefix, "spiking_self", address_space)
        self._update_stats(stats, address, response, candidate_mask)
        stats.q_rate.update(float(q_heads.float().mean().item()))
        stats.k_rate.update(float(k_heads.float().mean().item()))
        stats.gate_rate.update(float(_as_binary(buf.get("attn", proj)).float().mean().item()))
        stats.batches += 1

    def _update_stats(
        self,
        stats: ModuleDiagnostic,
        address: torch.Tensor,
        response: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> None:
        if self.record_callback is not None:
            self.record_callback(stats, address, response, candidate_mask)
        address_cpu = address.detach().to("cpu").tolist()
        response_cpu = response.detach().to("cpu").float().tolist()
        candidate_cpu = candidate_mask.detach().to("cpu").bool().tolist()
        stats.buckets.update_many(address_cpu, response_cpu)
        stats.response.update_many(response_cpu)
        candidate_values = [v for v, m in zip(response_cpu, candidate_cpu) if m]
        background_values = [v for v, m in zip(response_cpu, candidate_cpu) if not m]
        stats.candidate.update_many(candidate_values)
        stats.background.update_many(background_values)

    def summary(self) -> Dict[str, object]:
        modules = [stats.summary() for stats in self.module_stats.values()]
        stages = self._group_summary(modules, "stage")
        blocks = self._group_summary(modules, "name")
        return {
            "module_summary": modules,
            "per_stage_summary": stages,
            "per_block_summary": blocks,
            "address_coverage": self._mean_metric(modules, "address_coverage"),
            "bucket_occupancy": self._aggregate_occupancy(modules),
            "singleton_fraction": self._mean_metric(modules, "singleton_fraction"),
            "conditional_variance": self._mean_metric(modules, "conditional_variance"),
            "candidate_background_variance": {
                "candidate": self._mean_metric(modules, "candidate_variance"),
                "background": self._mean_metric(modules, "background_variance"),
            },
        }

    def _group_summary(self, modules: List[Dict[str, object]], key: str) -> Dict[str, Dict[str, float]]:
        grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for item in modules:
            grouped[str(item[key])].append(item)
        return {
            name: {
                "address_coverage": self._mean_metric(items, "address_coverage"),
                "singleton_fraction": self._mean_metric(items, "singleton_fraction"),
                "conditional_variance": self._mean_metric(items, "conditional_variance"),
                "candidate_variance": self._mean_metric(items, "candidate_variance"),
                "background_variance": self._mean_metric(items, "background_variance"),
                "num_modules": float(len(items)),
            }
            for name, items in grouped.items()
        }

    @staticmethod
    def _mean_metric(items: List[Dict[str, object]], key: str) -> Optional[float]:
        values = [float(item[key]) for item in items if item.get(key) is not None]
        if not values:
            return None
        return float(sum(values) / len(values))

    @staticmethod
    def _aggregate_occupancy(items: List[Dict[str, object]]) -> Dict[str, Optional[float]]:
        keys = ("min", "p50", "mean", "p90", "max")
        out: Dict[str, Optional[float]] = {}
        for key in keys:
            values = [
                float(item["bucket_occupancy"][key])
                for item in items
                if item.get("bucket_occupancy") is not None
            ]
            out[key] = float(sum(values) / len(values)) if values else None
        return out
