#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml

from qkformer_lut.hooks import ModuleDiagnostic, QKAddressDiagnostic
from qkformer_lut.stats import VarianceStats


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cifar10_model(root: Path, cfg: Dict[str, object]) -> torch.nn.Module:
    family = str(cfg.get("family", "cifar10"))
    if family not in {"cifar10", "cifar100", "cifar10dvs"}:
        raise ValueError(f"unsupported QKFormer model family: {family}")
    model_dir = root / ("cifar10-dvs" if family == "cifar10dvs" else family)
    sys.path.insert(0, str(model_dir))
    try:
        model_module = importlib.import_module("model")
        model_factory = getattr(model_module, "spiking_transformer", None)
        if model_factory is None:
            model_factory = getattr(model_module, "QKFormer")
        model = model_factory(
            drop_rate=0.0,
            drop_path_rate=0.0,
            img_size_h=int(cfg["img_size"]),
            img_size_w=int(cfg["img_size"]),
            patch_size=int(cfg["patch_size"]),
            embed_dims=int(cfg["dim"]),
            num_heads=int(cfg["num_heads"]),
            mlp_ratios=int(cfg["mlp_ratio"]),
            in_channels=int(cfg["in_channels"]),
            num_classes=int(cfg["num_classes"]),
            qkv_bias=False,
            depths=int(cfg["layer"]),
            sr_ratios=1,
            T=int(cfg["time_step"]),
        )
    finally:
        try:
            sys.path.remove(str(model_dir))
        except ValueError:
            pass
    return model


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key.replace("module.", "", 1): value for key, value in state.items()}


def load_checkpoint_if_available(model: torch.nn.Module, checkpoint: Optional[str]) -> Dict[str, object]:
    if not checkpoint:
        return {"loaded": False, "path": None, "missing_keys": None, "unexpected_keys": None}
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict):
        state = payload.get("state_dict") or payload.get("model") or payload
    else:
        state = payload
    state = _strip_module_prefix(state)
    msg = model.load_state_dict(state, strict=False)
    return {
        "loaded": True,
        "path": str(path),
        "missing_keys": list(msg.missing_keys),
        "unexpected_keys": list(msg.unexpected_keys),
    }


def make_cifar10_loader(
    data_dir: str,
    split: str,
    batch_size: int,
    workers: int,
    device: torch.device,
    shuffle: bool = False,
    seed: Optional[int] = None,
    index_start: int = 0,
    index_count: Optional[int] = None,
):
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    root = Path(data_dir).expanduser()
    dataset = datasets.CIFAR10(
        root=str(root),
        train=split == "train",
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        ),
    )
    if index_start or index_count is not None:
        start = max(0, int(index_start))
        stop = len(dataset) if index_count is None else min(len(dataset), start + int(index_count))
        if start >= stop:
            raise ValueError(f"empty CIFAR-10 subset: start={start}, stop={stop}, size={len(dataset)}")
        dataset = Subset(dataset, range(start, stop))
    generator = None
    if shuffle:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def make_cifar100_loader(
    data_dir: str,
    split: str,
    batch_size: int,
    workers: int,
    device: torch.device,
    shuffle: bool = False,
    seed: Optional[int] = None,
    index_start: int = 0,
    index_count: Optional[int] = None,
):
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    root = Path(data_dir).expanduser()
    dataset = datasets.CIFAR100(
        root=str(root),
        train=split == "train",
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        ),
    )
    if index_start or index_count is not None:
        start = max(0, int(index_start))
        stop = len(dataset) if index_count is None else min(len(dataset), start + int(index_count))
        if start >= stop:
            raise ValueError(f"empty CIFAR-100 subset: start={start}, stop={stop}, size={len(dataset)}")
        dataset = Subset(dataset, range(start, stop))
    generator = None
    if shuffle:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def make_cifar10dvs_loader(
    data_dir: str,
    split: str,
    batch_size: int,
    workers: int,
    device: torch.device,
    shuffle: bool = False,
    seed: Optional[int] = None,
    index_start: int = 0,
    index_count: Optional[int] = None,
):
    import math

    from spikingjelly.datasets import cifar10_dvs
    from torch.utils.data import DataLoader, Subset

    frames_number = int(os.environ.get("QKFORMER_LUT_TIME_STEP", "16"))
    dataset = cifar10_dvs.CIFAR10DVS(
        root=str(Path(data_dir).expanduser()),
        data_type="frame",
        frames_number=frames_number,
        split_by="number",
    )
    class_indices: Dict[int, List[int]] = defaultdict(list)
    for index, target in enumerate(dataset.targets):
        class_indices[int(target)].append(index)
    selected: List[int] = []
    use_train_split = split == "train"
    for target in sorted(class_indices):
        indices = class_indices[target]
        boundary = math.ceil(len(indices) * 0.9)
        selected.extend(indices[:boundary] if use_train_split else indices[boundary:])
    if index_start or index_count is not None:
        start = max(0, int(index_start))
        stop = len(selected) if index_count is None else min(len(selected), start + int(index_count))
        if start >= stop:
            raise ValueError(
                f"empty CIFAR10-DVS subset: start={start}, stop={stop}, size={len(selected)}"
            )
        selected = selected[start:stop]
    generator = None
    if shuffle:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))
    return DataLoader(
        Subset(dataset, selected),
        batch_size=batch_size,
        shuffle=bool(shuffle),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def make_timm_cifar_loader(
    mode: str,
    data_dir: str,
    split: str,
    batch_size: int,
    workers: int,
    device: torch.device,
):
    if split == "train":
        raise ValueError("the timm-compatible loader is reserved for deterministic evaluation")
    from timm.data import create_dataset, create_loader

    dataset_name = "torch/cifar100" if mode == "cifar100" else "torch/cifar10"
    dataset = create_dataset(
        dataset_name,
        root=str(Path(data_dir).expanduser()),
        split="validation",
        is_training=False,
        batch_size=batch_size,
    )
    return create_loader(
        dataset,
        input_size=(3, 32, 32),
        batch_size=batch_size,
        is_training=False,
        use_prefetcher=device.type == "cuda",
        interpolation="bicubic",
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        num_workers=workers,
        distributed=False,
        crop_pct=1.0,
        pin_memory=False,
    )


def build_loader(cfg: Dict[str, object], device: torch.device):
    mode = str(cfg.get("mode", "cifar10"))
    if str(cfg.get("backend", "torchvision")) == "timm":
        return make_timm_cifar_loader(
            mode=mode,
            data_dir=str(cfg["data_dir"]),
            split=str(cfg.get("split", "validation")),
            batch_size=int(cfg["batch_size"]),
            workers=int(cfg.get("workers", 4)),
            device=device,
        )
    loaders = {
        "cifar10": make_cifar10_loader,
        "cifar100": make_cifar100_loader,
        "cifar10dvs": make_cifar10dvs_loader,
    }
    if mode not in loaders:
        raise ValueError(f"E2/E3 requires data.mode in {sorted(loaders)}, got {mode}")
    return loaders[mode](
        data_dir=str(cfg["data_dir"]),
        split=str(cfg.get("split", "validation")),
        batch_size=int(cfg["batch_size"]),
        workers=int(cfg.get("workers", 4)),
        device=device,
        shuffle=bool(cfg.get("shuffle", False)),
        seed=int(cfg["seed"]) if cfg.get("seed") is not None else None,
        index_start=int(cfg.get("index_start", 0)),
        index_count=int(cfg["index_count"]) if cfg.get("index_count") is not None else None,
    )


def reset_model_state(model: torch.nn.Module) -> None:
    try:
        from spikingjelly.clock_driven import functional
    except Exception:
        return
    functional.reset_net(model)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> List[float]:
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        out = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            out.append(float(correct_k.mul_(100.0 / target.numel()).item()))
        return out


@dataclass
class AverageStats:
    count: int = 0
    total: float = 0.0

    def update(self, value: float, count: int) -> None:
        self.count += int(count)
        self.total += float(value) * int(count)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


@dataclass
class MSEStats:
    count: int = 0
    sse: float = 0.0

    def update(self, squared_error_sum: float, count: int) -> None:
        self.count += int(count)
        self.sse += float(squared_error_sum)

    @property
    def mse(self) -> float:
        return self.sse / self.count if self.count else 0.0


class ModulePrototype:
    def __init__(self, stats: ModuleDiagnostic, min_count: int) -> None:
        self.name = stats.name
        self.kind = stats.kind
        self.stage = stats.stage
        self.block = stats.block
        self.address_space = int(stats.address_space)
        self.min_count = int(min_count)
        self.count = torch.zeros(self.address_space, dtype=torch.float64)
        self.sum = torch.zeros(self.address_space, dtype=torch.float64)
        self.response = VarianceStats()
        self._address_mean: Optional[torch.Tensor] = None

    def update(self, address: torch.Tensor, response: torch.Tensor, _candidate_mask: torch.Tensor) -> None:
        addr = address.detach().to("cpu", dtype=torch.long)
        resp = response.detach().to("cpu", dtype=torch.float64)
        self.count += torch.bincount(addr, minlength=self.address_space).to(torch.float64)
        self.sum += torch.bincount(addr, weights=resp, minlength=self.address_space).to(torch.float64)
        self.response.update_many(resp.tolist())
        self._address_mean = None

    @property
    def global_mean(self) -> float:
        return self.response.mean

    @property
    def address_mean(self) -> torch.Tensor:
        if self._address_mean is None:
            means = torch.full_like(self.sum, fill_value=float(self.global_mean))
            seen = self.count > 0
            means[seen] = self.sum[seen] / self.count[seen]
            self._address_mean = means
        return self._address_mean

    def predict(self, address: torch.Tensor, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        addr = address.detach().to("cpu", dtype=torch.long)
        seen = self.count[addr] >= self.min_count
        pred = self.address_mean[addr]
        if not bool(seen.all().item()):
            global_pred = torch.full_like(pred, fill_value=float(self.global_mean))
            pred = torch.where(seen, pred, global_pred)
        return pred.to(device=device, dtype=dtype), seen.to(device=device)

    def summary(self) -> Dict[str, object]:
        seen = self.count > 0
        return {
            "name": self.name,
            "kind": self.kind,
            "stage": self.stage,
            "block": self.block,
            "address_space": self.address_space,
            "num_samples": int(self.count.sum().item()),
            "unique_addresses": int(seen.sum().item()),
            "address_coverage": float(seen.sum().item() / self.address_space) if self.address_space else None,
            "response_variance": self.response.variance,
        }


class PrototypeBank:
    def __init__(self, min_count: int) -> None:
        self.min_count = int(min_count)
        self.prototypes: Dict[str, ModulePrototype] = {}

    def calibrate(self, stats: ModuleDiagnostic, address: torch.Tensor, response: torch.Tensor, candidate_mask: torch.Tensor) -> None:
        proto = self.prototypes.get(stats.name)
        if proto is None:
            proto = ModulePrototype(stats, self.min_count)
            self.prototypes[stats.name] = proto
        proto.update(address, response, candidate_mask)

    def summary(self) -> Dict[str, Dict[str, object]]:
        return {name: proto.summary() for name, proto in self.prototypes.items()}


def _as_binary(x: torch.Tensor) -> torch.Tensor:
    return (x.detach() > 0).to(torch.long)


def _bin_index(values: torch.Tensor, bins: int, max_value: int) -> torch.Tensor:
    if max_value <= 0:
        return torch.zeros_like(values, dtype=torch.long)
    return torch.clamp((values.long() * bins) // max_value, min=0, max=bins - 1)


def _population_bin(popcount: torch.Tensor, bins: int, depth: int) -> torch.Tensor:
    if depth <= 0:
        return torch.zeros_like(popcount, dtype=torch.long)
    return torch.clamp((popcount.long() * bins) // (depth + 1), min=0, max=bins - 1)


def _flatten_proj(output: torch.Tensor, time_steps: int, batch_size: int) -> torch.Tensor:
    if output.ndim == 5:
        t, b, c, h, w = output.shape
        return output.reshape(t, b, c, h * w)
    if output.ndim == 4:
        return output
    if output.ndim == 3:
        tb, c, n = output.shape
        if tb != time_steps * batch_size:
            raise ValueError(
                "Projected output batch mismatch: "
                f"shape={tuple(output.shape)}, T={time_steps}, B={batch_size}"
            )
        return output.reshape(time_steps, batch_size, c, n)
    raise ValueError(f"Unsupported projected output shape: {tuple(output.shape)}")


def token_qk_full_address(
    q_output: torch.Tensor,
    k_output: torch.Tensor,
    attn_output: torch.Tensor,
    proj_output: torch.Tensor,
    token_bins: int,
    channel_bins: int,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int, int], str]:
    q_raw = _as_binary(q_output)
    k_raw = _as_binary(k_output)
    gate_raw = _as_binary(attn_output)
    t, b, c, n = q_raw.shape
    proj = _flatten_proj(proj_output, t, b).float()
    heads = gate_raw.shape[2]
    depth = c // heads
    k_heads = k_raw.reshape(t, b, heads, depth, n)
    gate = gate_raw.reshape(t, b, heads, 1, n)
    token_idx = torch.arange(n, device=q_raw.device).reshape(1, 1, 1, 1, n).expand(t, b, heads, depth, n)
    channel_idx = torch.arange(depth, device=q_raw.device).reshape(1, 1, 1, depth, 1).expand(t, b, heads, depth, n)
    head_idx = torch.arange(heads, device=q_raw.device).reshape(1, 1, heads, 1, 1).expand(t, b, heads, depth, n)
    q_gate = gate.expand(t, b, heads, depth, n)
    k_bit = k_heads
    token_bin = _bin_index(token_idx, token_bins, n)
    channel_bin = _bin_index(channel_idx, channel_bins, depth)
    address = (((head_idx * token_bins + token_bin) * channel_bins + channel_bin) * 2 + q_gate) * 2 + k_bit
    response = proj.reshape(t, b, heads, depth, n)
    return address.reshape(-1), response.reshape(-1), (t, b, c, n), "token_qk"


def spiking_self_full_address(
    module: torch.nn.Module,
    q_output: torch.Tensor,
    k_output: torch.Tensor,
    proj_output: torch.Tensor,
    token_bins: int,
    channel_bins: int,
    population_bins: int,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int, int], str]:
    q_raw = _as_binary(q_output)
    k_raw = _as_binary(k_output)
    t, b, c, n = q_raw.shape
    proj = _flatten_proj(proj_output, t, b).float()
    heads = int(getattr(module, "num_heads", 1))
    depth = c // heads
    q_heads = q_raw.transpose(-1, -2).reshape(t, b, n, heads, depth).permute(0, 1, 3, 2, 4)
    k_heads = k_raw.transpose(-1, -2).reshape(t, b, n, heads, depth).permute(0, 1, 3, 2, 4)
    q_pop = q_heads.sum(dim=-1)
    k_pop = k_heads.sum(dim=-1)
    token_idx = torch.arange(n, device=q_raw.device).reshape(1, 1, 1, n, 1).expand(t, b, heads, n, depth)
    channel_idx = torch.arange(depth, device=q_raw.device).reshape(1, 1, 1, 1, depth).expand(t, b, heads, n, depth)
    head_idx = torch.arange(heads, device=q_raw.device).reshape(1, 1, heads, 1, 1).expand(t, b, heads, n, depth)
    q_bit = q_heads
    k_bit = k_heads
    q_pop_bin = _population_bin(q_pop, population_bins, depth).unsqueeze(-1).expand(t, b, heads, n, depth)
    k_pop_bin = _population_bin(k_pop, population_bins, depth).unsqueeze(-1).expand(t, b, heads, n, depth)
    token_bin = _bin_index(token_idx, token_bins, n)
    channel_bin = _bin_index(channel_idx, channel_bins, depth)
    address = head_idx
    address = address * token_bins + token_bin
    address = address * channel_bins + channel_bin
    address = address * 2 + q_bit
    address = address * 2 + k_bit
    address = address * population_bins + q_pop_bin
    address = address * population_bins + k_pop_bin
    response = proj.reshape(t, b, heads, depth, n).permute(0, 1, 2, 4, 3)
    return address.reshape(-1), response.reshape(-1), (t, b, c, n), "spiking_self"


def restore_token_prediction(pred: torch.Tensor, layout: Tuple[int, int, int, int], original: torch.Tensor) -> torch.Tensor:
    t, b, c, n = layout
    pred_flat = pred.reshape(t, b, c, n)
    return pred_flat.reshape_as(original)


def restore_spiking_prediction(pred: torch.Tensor, layout: Tuple[int, int, int, int], original: torch.Tensor, heads: int) -> torch.Tensor:
    t, b, c, n = layout
    depth = c // heads
    pred_flat = pred.reshape(t, b, heads, n, depth).permute(0, 1, 2, 4, 3).reshape(t, b, c, n)
    return pred_flat.reshape_as(original)


class ReplacementStats:
    def __init__(self, name: str, kind: str, stage: str) -> None:
        self.name = name
        self.kind = kind
        self.stage = stage
        self.local_mse = MSEStats()
        self.hit_count = 0
        self.total = 0

    def update(self, original: torch.Tensor, replacement: torch.Tensor, seen: torch.Tensor) -> None:
        orig = original.detach().float()
        repl = replacement.detach().float()
        self.local_mse.update(float(torch.sum((orig - repl) ** 2).item()), int(orig.numel()))
        self.hit_count += int(seen.sum().item())
        self.total += int(seen.numel())

    def summary(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "stage": self.stage,
            "local_replacement_mse": self.local_mse.mse,
            "address_hit_rate": self.hit_count / float(self.total) if self.total else 0.0,
            "num_replaced_values": self.total,
        }


class QKProjectionReplacer:
    def __init__(
        self,
        model: torch.nn.Module,
        prototypes: Dict[str, ModulePrototype],
        target_modules: Iterable[str],
        token_bins: int,
        channel_bins: int,
        population_bins: int,
        blend: float = 1.0,
        mode: str = "address_lut",
        mode_seed: int = 0,
    ) -> None:
        self.model = model
        self.prototypes = prototypes
        self.targets = set(target_modules)
        self.token_bins = int(token_bins)
        self.channel_bins = int(channel_bins)
        self.population_bins = int(population_bins)
        self.blend = float(blend)
        self.mode = str(mode)
        if self.mode not in {"address_lut", "global_mean", "shuffled_address_lut"}:
            raise ValueError(f"unsupported replacement mode: {self.mode}")
        self.mode_seed = int(mode_seed)
        self.enabled = False
        self.buffers: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.modules = dict(model.named_modules())
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.stats: Dict[str, ReplacementStats] = {}
        self._address_permutations: Dict[str, torch.Tensor] = {}
        self._register()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _register(self) -> None:
        for name, module in self.modules.items():
            if name not in self.targets:
                continue
            if name not in self.prototypes:
                continue
            for child_name in ("q_lif", "k_lif", "attn_lif", "proj_lif"):
                child = getattr(module, child_name, None)
                if child is None:
                    continue
                self.handles.append(child.register_forward_hook(self._make_hook(name, child_name.replace("_lif", ""))))

    def _make_hook(self, prefix: str, kind: str):
        def hook(_module, _inputs, output):
            self.buffers[prefix][kind] = output.detach()
            if kind != "proj":
                return None
            try:
                if not self.enabled:
                    return None
                return self._replace(prefix, output)
            finally:
                self.buffers[prefix].clear()

        return hook

    def _replace(self, prefix: str, output: torch.Tensor) -> Optional[torch.Tensor]:
        buf = self.buffers[prefix]
        if "q" not in buf or "k" not in buf:
            return None
        proto = self.prototypes.get(prefix)
        if proto is None:
            return None
        module = self.modules[prefix]
        cls_name = module.__class__.__name__
        if cls_name == "Token_QK_Attention" and "attn" in buf:
            address, response, layout, kind = token_qk_full_address(
                buf["q"], buf["k"], buf["attn"], output, self.token_bins, self.channel_bins
            )
            response_device = response.to(device=output.device, dtype=output.dtype)
            pred, seen = self._predict(proto, address, response_device)
            blended = response_device * (1.0 - self.blend) + pred * self.blend
            replacement = restore_token_prediction(blended, layout, output)
        elif cls_name == "Spiking_Self_Attention":
            address, response, layout, kind = spiking_self_full_address(
                module,
                buf["q"],
                buf["k"],
                output,
                self.token_bins,
                self.channel_bins,
                self.population_bins,
            )
            response_device = response.to(device=output.device, dtype=output.dtype)
            pred, seen = self._predict(proto, address, response_device)
            blended = response_device * (1.0 - self.blend) + pred * self.blend
            replacement = restore_spiking_prediction(blended, layout, output, int(getattr(module, "num_heads", 1)))
        else:
            return None
        stat = self.stats.get(prefix)
        if stat is None:
            stat = ReplacementStats(prefix, kind, proto.stage)
            self.stats[prefix] = stat
        stat.update(response, blended, seen)
        return replacement

    def _predict(
        self,
        proto: ModulePrototype,
        address: torch.Tensor,
        response_device: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "global_mean":
            pred = torch.full_like(response_device, fill_value=float(proto.global_mean))
            seen = torch.ones_like(address, dtype=torch.bool, device=response_device.device)
            return pred, seen
        if self.mode == "shuffled_address_lut":
            perm = self._address_permutation(proto)
            shuffled = perm[address.detach().to("cpu", dtype=torch.long)]
            return proto.predict(shuffled, response_device.device, response_device.dtype)
        return proto.predict(address, response_device.device, response_device.dtype)

    def _address_permutation(self, proto: ModulePrototype) -> torch.Tensor:
        perm = self._address_permutations.get(proto.name)
        if perm is None:
            digest = hashlib.sha256(f"{self.mode_seed}:{proto.name}".encode("utf-8")).hexdigest()
            seed = int(digest[:16], 16) % (2**63)
            generator = torch.Generator()
            generator.manual_seed(seed)
            perm = torch.randperm(proto.address_space, generator=generator, dtype=torch.long)
            self._address_permutations[proto.name] = perm
        return perm

    def summary(self) -> Dict[str, Dict[str, object]]:
        return {name: stat.summary() for name, stat in self.stats.items()}


def calibrate_prototypes(model, loader, data_cfg, diag_cfg, bank, device) -> Tuple[int, Dict[str, object]]:
    diagnostic = QKAddressDiagnostic(
        model,
        token_bins=int(diag_cfg.get("token_bins", 8)),
        channel_bins=int(diag_cfg.get("channel_bins", 8)),
        population_bins=int(diag_cfg.get("population_bins", 4)),
        max_records_per_module_per_batch=int(diag_cfg.get("max_records_per_module_per_batch", 65536)),
        response_child=str(diag_cfg.get("response_child", "proj_lif")),
        record_callback=bank.calibrate,
    )
    max_batches = int(data_cfg["num_batches"])
    processed = 0
    with torch.no_grad():
        for images, _targets in loader:
            if max_batches > 0 and processed >= max_batches:
                break
            images = images.to(device, non_blocking=True).float()
            _ = model(images)
            reset_model_state(model)
            processed += 1
    summary = diagnostic.summary()
    diagnostic.close()
    return processed, summary


class EvalSummary:
    def __init__(self) -> None:
        self.loss = AverageStats()
        self.top1 = AverageStats()
        self.top5 = AverageStats()
        self.logit_mse = MSEStats()
        self.kl = AverageStats()

    def model_dict(self) -> Dict[str, float]:
        return {
            "loss": self.loss.mean,
            "top1": self.top1.mean,
            "top5": self.top5.mean,
        }


def evaluate_replacement(model, loader, data_cfg, replacer, device) -> Tuple[int, Dict[str, object]]:
    loss_fn = nn.CrossEntropyLoss().to(device)
    baseline = EvalSummary()
    replacement = EvalSummary()
    max_batches = int(data_cfg["num_batches"])
    processed = 0
    with torch.no_grad():
        for images, targets in loader:
            if max_batches > 0 and processed >= max_batches:
                break
            images = images.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True)
            batch_size = int(targets.numel())

            replacer.enabled = False
            baseline_logits = model(images)
            baseline_loss = loss_fn(baseline_logits, targets)
            b_top1, b_top5 = accuracy(baseline_logits, targets)
            baseline.loss.update(float(baseline_loss.item()), batch_size)
            baseline.top1.update(b_top1, batch_size)
            baseline.top5.update(b_top5, batch_size)
            reset_model_state(model)

            replacer.enabled = True
            replacement_logits = model(images)
            replacement_loss = loss_fn(replacement_logits, targets)
            r_top1, r_top5 = accuracy(replacement_logits, targets)
            replacement.loss.update(float(replacement_loss.item()), batch_size)
            replacement.top1.update(r_top1, batch_size)
            replacement.top5.update(r_top5, batch_size)
            replacement.logit_mse.update(
                float(torch.sum((replacement_logits.detach() - baseline_logits.detach()) ** 2).item()),
                int(baseline_logits.numel()),
            )
            kl = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(replacement_logits, dim=1),
                torch.nn.functional.softmax(baseline_logits, dim=1),
                reduction="batchmean",
            )
            replacement.kl.update(float(kl.item()), batch_size)
            reset_model_state(model)
            processed += 1

    replacer.enabled = False
    base = baseline.model_dict()
    repl = replacement.model_dict()
    return processed, {
        "baseline": base,
        "replacement": {
            **repl,
            "logit_mse": replacement.logit_mse.mse,
            "kl_to_baseline": replacement.kl.mean,
        },
        "delta": {
            "loss": repl["loss"] - base["loss"],
            "top1": repl["top1"] - base["top1"],
            "top5": repl["top5"] - base["top5"],
        },
    }


def run(config_path: Path, output_dir: Path) -> Dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(config_path)
    set_seed(int(cfg["experiment"].get("seed", 42)))
    model_cfg = dict(cfg["model"])
    diag_cfg = dict(cfg["diagnostic"])
    data_cfg = dict(cfg["data"])
    calibration_cfg = dict(data_cfg["calibration"])
    evaluation_cfg = dict(data_cfg["evaluation"])
    replacement_cfg = dict(cfg["replacement"])

    env_checkpoint = os.environ.get("QKFORMER_LUT_CKPT")
    if env_checkpoint:
        model_cfg["checkpoint"] = env_checkpoint
    env_time_step = os.environ.get("QKFORMER_LUT_TIME_STEP")
    if env_time_step:
        model_cfg["time_step"] = int(env_time_step)
    env_data_dir = os.environ.get("QKFORMER_LUT_DATA_DIR")
    if env_data_dir:
        calibration_cfg["data_dir"] = env_data_dir
        evaluation_cfg["data_dir"] = env_data_dir
    env_calib_batches = os.environ.get("QKFORMER_LUT_E2_CALIB_BATCHES")
    if env_calib_batches:
        calibration_cfg["num_batches"] = int(env_calib_batches)
    env_calib_shuffle = os.environ.get("QKFORMER_LUT_E2_CALIB_SHUFFLE")
    if env_calib_shuffle:
        calibration_cfg["shuffle"] = env_calib_shuffle.strip().lower() in {"1", "true", "yes", "on"}
    env_calib_seed = os.environ.get("QKFORMER_LUT_E2_CALIB_SEED")
    if env_calib_seed:
        calibration_cfg["seed"] = int(env_calib_seed)
    env_eval_batches = os.environ.get("QKFORMER_LUT_E2_EVAL_BATCHES")
    if env_eval_batches:
        evaluation_cfg["num_batches"] = int(env_eval_batches)
    env_targets = os.environ.get("QKFORMER_LUT_E2_TARGETS")
    if env_targets:
        replacement_cfg["target_modules"] = [
            item.strip() for item in env_targets.split(",") if item.strip()
        ]
    env_blend = os.environ.get("QKFORMER_LUT_E2_BLEND")
    if env_blend:
        replacement_cfg["blend"] = float(env_blend)
    env_mode = os.environ.get("QKFORMER_LUT_E2_MODE")
    if env_mode:
        replacement_cfg["mode"] = env_mode
    env_mode_seed = os.environ.get("QKFORMER_LUT_E2_MODE_SEED")
    if env_mode_seed:
        replacement_cfg["mode_seed"] = int(env_mode_seed)

    device_name = str(diag_cfg.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the upstream cupy-backed QKFormer LIF modules")
    device = torch.device(device_name)
    model = build_cifar10_model(root, model_cfg)
    checkpoint_info = load_checkpoint_if_available(model, model_cfg.get("checkpoint"))
    model.to(device)
    model.eval()

    calibration_loader = build_loader(calibration_cfg, device)
    evaluation_loader = build_loader(evaluation_cfg, device)
    bank = PrototypeBank(min_count=int(diag_cfg.get("prototype_min_count", 2)))
    print("[qk-lut-e2] calibration_start")
    calibration_batches, calibration_hook_summary = calibrate_prototypes(
        model, calibration_loader, calibration_cfg, diag_cfg, bank, device
    )
    print(f"[qk-lut-e2] calibration_batches={calibration_batches}")

    target_modules = list(replacement_cfg.get("target_modules", []))
    if not target_modules:
        target_modules = ["stage1.0.tssa", "stage2.0.tssa"]
    replacer = QKProjectionReplacer(
        model,
        bank.prototypes,
        target_modules,
        token_bins=int(diag_cfg.get("token_bins", 8)),
        channel_bins=int(diag_cfg.get("channel_bins", 8)),
        population_bins=int(diag_cfg.get("population_bins", 4)),
        blend=float(replacement_cfg.get("blend", 1.0)),
        mode=str(replacement_cfg.get("mode", "address_lut")),
        mode_seed=int(replacement_cfg.get("mode_seed", 0)),
    )
    print(f"[qk-lut-e2] target_modules={target_modules}")
    print("[qk-lut-e2] replacement_eval_start")
    evaluation_batches, replacement_summary = evaluate_replacement(
        model, evaluation_loader, evaluation_cfg, replacer, device
    )
    local_summary = replacer.summary()
    replacer.close()

    verdict = "PENDING_PHASE_GATE_REVIEW"
    if not checkpoint_info["loaded"]:
        verdict = "PENDING_REAL_DATA_CHECKPOINT"
    metrics = {
        "experiment": cfg["experiment"],
        "model": {
            "family": model_cfg["family"],
            "time_step": model_cfg["time_step"],
            "layer": model_cfg["layer"],
            "dim": model_cfg["dim"],
            "num_heads": model_cfg["num_heads"],
            "checkpoint": checkpoint_info,
        },
        "data": {
            "calibration": {
                "source": "cifar10",
                "split": calibration_cfg.get("split"),
                "num_batches": calibration_batches,
                "batch_size": calibration_cfg["batch_size"],
                "shuffle": bool(calibration_cfg.get("shuffle", False)),
                "seed": calibration_cfg.get("seed"),
            },
            "evaluation": {
                "source": "cifar10",
                "split": evaluation_cfg.get("split"),
                "num_batches": evaluation_batches,
                "batch_size": evaluation_cfg["batch_size"],
            },
        },
        "diagnostic_config": diag_cfg,
        "replacement_config": replacement_cfg,
        "env_overrides": {
            "QKFORMER_LUT_E2_CALIB_BATCHES": env_calib_batches,
            "QKFORMER_LUT_TIME_STEP": env_time_step,
            "QKFORMER_LUT_E2_CALIB_SHUFFLE": env_calib_shuffle,
            "QKFORMER_LUT_E2_CALIB_SEED": env_calib_seed,
            "QKFORMER_LUT_E2_EVAL_BATCHES": env_eval_batches,
            "QKFORMER_LUT_E2_TARGETS": env_targets,
            "QKFORMER_LUT_E2_BLEND": env_blend,
            "QKFORMER_LUT_E2_MODE": env_mode,
            "QKFORMER_LUT_E2_MODE_SEED": env_mode_seed,
        },
        "target_modules": target_modules,
        "verdict": verdict,
        "classification": replacement_summary,
        "local_replacement": local_summary,
        "calibration_prototypes": bank.summary(),
        "calibration_hook_summary": calibration_hook_summary,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(f"[qk-lut-e2] wrote {metrics_path}")
    print(f"[qk-lut-e2] verdict={verdict}")
    print(f"[qk-lut-e2] baseline_top1={replacement_summary['baseline']['top1']}")
    print(f"[qk-lut-e2] replacement_top1={replacement_summary['replacement']['top1']}")
    print(f"[qk-lut-e2] delta_top1={replacement_summary['delta']['top1']}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="QK-LUTFormer E2 stage-wise replacement diagnostic")
    parser.add_argument("--config", type=Path, default=Path("configs/qkformer_lut_e2_replace.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
