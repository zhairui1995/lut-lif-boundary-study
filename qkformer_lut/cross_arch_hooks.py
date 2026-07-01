from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LUTRunMetrics:
    zero_count: int = 0
    value_count: int = 0
    hit_count: int = 0
    lookup_count: int = 0
    scalar_entries: int = 0
    value_bytes: int = 4

    def update_sparsity(self, x: torch.Tensor) -> None:
        binary = x.detach() > 0
        self.zero_count += int((~binary).sum().item())
        self.value_count += int(binary.numel())

    def update_lookup(self, hit_mask: torch.Tensor) -> None:
        hit = hit_mask.detach().bool()
        self.hit_count += int(hit.sum().item())
        self.lookup_count += int(hit.numel())

    def update_entries(self, entries: int, values_per_entry: int = 1) -> None:
        self.scalar_entries = int(entries) * int(values_per_entry)

    def summary(self) -> Dict[str, float]:
        hit_rate = self.hit_count / self.lookup_count if self.lookup_count else 0.0
        return {
            "spike_sparsity": self.zero_count / self.value_count if self.value_count else 0.0,
            "lut_coverage_hit_rate": hit_rate,
            "fallback_rate": 1.0 - hit_rate,
            "analytical_memory_footprint_kib": self.scalar_entries * self.value_bytes / 1024.0,
            "supported_scalar_entries": int(self.scalar_entries),
        }


class DensePrototypeLUT:
    """Small skeleton LUT for aligned/shuffled/token-channel controls.

    The calibration script should fill `sum` and `count` from a frozen backbone
    pass, then call `finalize()`. The evaluation hook calls `predict()` and
    falls back to either a token/channel mean table or a global mean.
    """

    def __init__(self, address_space: int, mode: str, min_support: int = 2, seed: int = 42) -> None:
        if mode not in {"aligned_lut", "shuffled_address", "token_channel_mean"}:
            raise ValueError(f"unsupported LUT mode: {mode}")
        self.address_space = int(address_space)
        self.mode = str(mode)
        self.min_support = int(min_support)
        self.seed = int(seed)
        self.sum = torch.zeros(self.address_space, dtype=torch.float64)
        self.count = torch.zeros(self.address_space, dtype=torch.float64)
        self.mean: Optional[torch.Tensor] = None
        self.support: Optional[torch.Tensor] = None
        self.global_mean = torch.tensor(0.0, dtype=torch.float32)
        self._perm: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update(self, address: torch.Tensor, response: torch.Tensor) -> None:
        addr = address.detach().reshape(-1).to("cpu", dtype=torch.long)
        resp = response.detach().reshape(-1).to("cpu", dtype=torch.float64)
        self.sum += torch.bincount(addr, weights=resp, minlength=self.address_space)
        self.count += torch.bincount(addr, minlength=self.address_space).to(torch.float64)
        if resp.numel():
            n_old = float(self.count.sum().item() - resp.numel())
            n_new = float(resp.numel())
            old_mean = float(self.global_mean.item())
            self.global_mean = torch.tensor((old_mean * n_old + float(resp.sum().item())) / max(n_old + n_new, 1.0))

    def finalize(self) -> None:
        support = self.count >= self.min_support
        mean = torch.full((self.address_space,), float(self.global_mean.item()), dtype=torch.float32)
        valid = self.count > 0
        mean[valid] = (self.sum[valid] / self.count[valid]).to(torch.float32)
        self.mean = mean
        self.support = support

    def _address_for_mode(self, address: torch.Tensor) -> torch.Tensor:
        if self.mode != "shuffled_address":
            return address
        if self._perm is None:
            gen = torch.Generator(device="cpu").manual_seed(self.seed)
            self._perm = torch.randperm(self.address_space, generator=gen, dtype=torch.long)
        return self._perm[address.detach().to("cpu", dtype=torch.long)].to(address.device)

    @torch.no_grad()
    def predict(self, address: torch.Tensor, fallback: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mean is None or self.support is None:
            raise RuntimeError("call finalize() before predict()")
        query = self._address_for_mode(address)
        cpu_query = query.detach().to("cpu", dtype=torch.long)
        pred = self.mean[cpu_query].to(address.device)
        seen = self.support[cpu_query].to(address.device)
        if fallback is None:
            fallback = torch.full_like(pred, float(self.global_mean.item()))
        pred = torch.where(seen, pred, fallback.reshape_as(pred))
        return pred, seen

    @property
    def supported_entries(self) -> int:
        if self.support is None:
            return int((self.count >= self.min_support).sum().item())
        return int(self.support.sum().item())


def freeze_backbone(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _bin_index(values: torch.Tensor, bins: int, max_value: int) -> torch.Tensor:
    if max_value <= 1:
        return torch.zeros_like(values, dtype=torch.long)
    return torch.clamp((values.long() * bins) // max_value, min=0, max=bins - 1)


def _population_bin(popcount: torch.Tensor, bins: int, depth: int) -> torch.Tensor:
    if depth <= 0:
        return torch.zeros_like(popcount, dtype=torch.long)
    return torch.clamp((popcount.long() * bins) // (depth + 1), min=0, max=bins - 1)


def _spikformer_to_tbnc(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, str, Tuple[int, ...]]:
    if x.ndim == 4 and x.shape[-1] == dim:
        return x, "T_B_N_C", tuple(x.shape)
    if x.ndim == 4 and x.shape[2] == dim:
        return x.transpose(2, 3).contiguous(), "T_B_C_N", tuple(x.shape)
    if x.ndim == 5:
        t, b, c, h, w = x.shape
        if c != dim:
            raise ValueError(f"expected C={dim} in 5D Spikformer tensor, got {tuple(x.shape)}")
        return x.flatten(3).transpose(2, 3).contiguous(), "T_B_C_H_W", tuple(x.shape)
    raise ValueError(f"unsupported Spikformer SSA tensor shape: {tuple(x.shape)}")


def _restore_spikformer(pred_tbnc: torch.Tensor, layout: str, shape: Tuple[int, ...]) -> torch.Tensor:
    if layout == "T_B_N_C":
        return pred_tbnc.reshape(shape)
    if layout == "T_B_C_N":
        return pred_tbnc.transpose(2, 3).reshape(shape)
    if layout == "T_B_C_H_W":
        t, b, c, h, w = shape
        return pred_tbnc.transpose(2, 3).reshape(t, b, c, h, w)
    raise ValueError(f"unknown layout: {layout}")


def spikformer_qk_scalar_address(
    q_tbnc: torch.Tensor,
    k_tbnc: torch.Tensor,
    num_heads: int,
    token_bins: int,
    channel_bins: int,
    population_bins: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    q_bin = (q_tbnc.detach() > 0).to(torch.long)
    k_bin = (k_tbnc.detach() > 0).to(torch.long)
    t, b, n, c = q_bin.shape
    heads = int(num_heads)
    depth = c // heads
    qh = q_bin.reshape(t, b, n, heads, depth).permute(0, 1, 3, 2, 4)
    kh = k_bin.reshape(t, b, n, heads, depth).permute(0, 1, 3, 2, 4)

    token_idx = torch.arange(n, device=q_bin.device).view(1, 1, 1, n, 1).expand(t, b, heads, n, depth)
    channel_idx = torch.arange(depth, device=q_bin.device).view(1, 1, 1, 1, depth).expand(t, b, heads, n, depth)
    head_idx = torch.arange(heads, device=q_bin.device).view(1, 1, heads, 1, 1).expand(t, b, heads, n, depth)

    q_pop = qh.sum(dim=-1, keepdim=True).expand_as(qh)
    k_pop = kh.sum(dim=-1, keepdim=True).expand_as(kh)
    token_bin = _bin_index(token_idx, token_bins, n)
    channel_bin = _bin_index(channel_idx, channel_bins, depth)
    coarse_address = (head_idx * token_bins + token_bin) * channel_bins + channel_bin
    address = coarse_address
    address = address * 2 + qh
    address = address * 2 + kh
    address = address * population_bins + _population_bin(q_pop, population_bins, depth)
    address = address * population_bins + _population_bin(k_pop, population_bins, depth)
    address = address.permute(0, 1, 3, 2, 4).reshape(t, b, n, c)
    coarse_address = coarse_address.permute(0, 1, 3, 2, 4).reshape(t, b, n, c)
    address_space = heads * token_bins * channel_bins * 4 * population_bins * population_bins
    coarse_address_space = heads * token_bins * channel_bins
    return address, coarse_address, address_space, coarse_address_space


class SpikformerSSALUTHook:
    """Track A skeleton for ZK-Zhou/spikformer SSA modules.

    Hooks the official SSA child modules named `q_lif`, `k_lif`, and
    `proj_lif`. During calibration it records response prototypes. During
    evaluation it returns a replacement tensor from `proj_lif`'s forward hook.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        mode: str = "aligned_lut",
        token_bins: int = 8,
        channel_bins: int = 8,
        population_bins: int = 4,
        min_support: int = 2,
        replace: bool = False,
        seed: int = 42,
    ) -> None:
        self.model = freeze_backbone(model)
        self.mode = mode
        self.token_bins = int(token_bins)
        self.channel_bins = int(channel_bins)
        self.population_bins = int(population_bins)
        self.min_support = int(min_support)
        self.replace = bool(replace)
        self.seed = int(seed)
        self.buffers: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.luts: Dict[str, DensePrototypeLUT] = {}
        self.metrics: Dict[str, LUTRunMetrics] = defaultdict(LUTRunMetrics)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def finalize(self) -> None:
        for name, lut in self.luts.items():
            lut.finalize()
            self.metrics[name].update_entries(lut.supported_entries)

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {name: metric.summary() for name, metric in self.metrics.items()}

    def _register(self) -> None:
        for name, module in self.model.named_modules():
            if module.__class__.__name__ != "SSA":
                continue
            for child_name in ("q_lif", "k_lif", "proj_lif"):
                child = getattr(module, child_name, None)
                if child is None:
                    continue
                self.handles.append(child.register_forward_hook(self._make_hook(name, child_name[:-4])))

    def _make_hook(self, prefix: str, kind: str):
        def hook(_module, _inputs, output):
            self.buffers[prefix][kind] = output.detach()
            if kind == "q":
                self.metrics[prefix].update_sparsity(output)
            elif kind == "k":
                self.metrics[prefix].update_sparsity(output)
            elif kind == "proj":
                return self._consume(prefix, output)
            return None

        return hook

    @torch.no_grad()
    def _consume(self, prefix: str, proj_output: torch.Tensor) -> Optional[torch.Tensor]:
        module = self.model.get_submodule(prefix)
        buf = self.buffers[prefix]
        if "q" not in buf or "k" not in buf:
            return None
        dim = int(getattr(module, "dim"))
        q_tbnc, _, _ = _spikformer_to_tbnc(buf["q"], dim)
        k_tbnc, _, _ = _spikformer_to_tbnc(buf["k"], dim)
        proj_tbnc, layout, shape = _spikformer_to_tbnc(proj_output, dim)
        address, coarse_address, address_space, coarse_address_space = spikformer_qk_scalar_address(
            q_tbnc=q_tbnc,
            k_tbnc=k_tbnc,
            num_heads=int(getattr(module, "num_heads")),
            token_bins=self.token_bins,
            channel_bins=self.channel_bins,
            population_bins=self.population_bins,
        )
        if self.mode == "token_channel_mean":
            address = coarse_address
            address_space = coarse_address_space
        lut = self.luts.get(prefix)
        if lut is None:
            lut = DensePrototypeLUT(address_space, mode=self.mode, min_support=self.min_support, seed=self.seed)
            self.luts[prefix] = lut
        if not self.replace:
            lut.update(address, proj_tbnc)
            return None
        pred, seen = lut.predict(address)
        self.metrics[prefix].update_lookup(seen)
        return _restore_spikformer(pred.reshape_as(proj_tbnc), layout, shape).to(proj_output.dtype)


def sew_local_conv_address(
    spike: torch.Tensor,
    out_channels: int,
    kernel_size: int,
    padding: int,
    spatial_bins: int,
    channel_bins: int,
    population_bins: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    if spike.ndim != 5:
        raise ValueError(f"expected SEW spike tensor [T,B,C,H,W], got {tuple(spike.shape)}")
    x = (spike.detach() > 0).to(torch.float32)
    t, b, c, h, w = x.shape
    flat = x.flatten(0, 1)
    patches = F.unfold(flat, kernel_size=kernel_size, padding=padding)
    patches = patches.transpose(1, 2).reshape(t, b, h * w, c, kernel_size * kernel_size)
    pop = patches.sum(dim=-1).to(torch.long)
    center_bit = patches[..., (kernel_size * kernel_size) // 2].to(torch.long)

    y = torch.arange(h, device=spike.device).view(h, 1).expand(h, w).reshape(-1)
    xidx = torch.arange(w, device=spike.device).view(1, w).expand(h, w).reshape(-1)
    spatial = (_bin_index(y, spatial_bins, h) * spatial_bins + _bin_index(xidx, spatial_bins, w))
    spatial = spatial.view(1, 1, h * w, 1).expand(t, b, h * w, c)
    channel = torch.arange(c, device=spike.device).view(1, 1, 1, c).expand(t, b, h * w, c)

    coarse = spatial * channel_bins + _bin_index(channel, channel_bins, c)
    base = coarse
    base = base * 2 + center_bit
    base = base * population_bins + _population_bin(pop, population_bins, kernel_size * kernel_size)

    # Conv2 maps input channels to output channels. The skeleton reuses the
    # same local input address for each output channel and appends output-channel
    # bins, so output shape matches sn2: [T,B,Cout,H,W].
    out_ch = torch.arange(out_channels, device=spike.device).view(1, 1, out_channels, 1, 1)
    base = base.permute(0, 1, 3, 2).reshape(t, b, c, h, w)
    if c != out_channels:
        base = base[:, :, :1].expand(t, b, out_channels, h, w)
    address = base + 0 * out_ch
    address_space = spatial_bins * spatial_bins * channel_bins * 2 * population_bins
    coarse = coarse.permute(0, 1, 3, 2).reshape(t, b, c, h, w)
    if c != out_channels:
        coarse = coarse[:, :, :1].expand(t, b, out_channels, h, w)
    coarse_address = coarse + 0 * out_ch
    coarse_address_space = spatial_bins * spatial_bins * channel_bins
    return address.long(), coarse_address.long(), address_space, coarse_address_space


class SEWResNetBasicBlockLUTHook:
    """Track B skeleton for fangwei123456/Spike-Element-Wise-ResNet.

    Targets ImageNet-style `sew_resnet.BasicBlock`:
        out = sn1(conv1(x)); out = sn2(conv2(out)); out = connect(out, identity)

    The hook stores the lossless `sn1` spike tensor produced by SpikingJelly's
    `MultiStepIFNode` and replaces the `sn2` spike tensor. That keeps the
    repository's residual ADD/AND/IAND routing unchanged.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        mode: str = "aligned_lut",
        spatial_bins: int = 8,
        channel_bins: int = 8,
        population_bins: int = 4,
        kernel_size: int = 3,
        padding: int = 1,
        min_support: int = 2,
        replace: bool = False,
        seed: int = 42,
    ) -> None:
        self.model = freeze_backbone(model)
        self.mode = mode
        self.spatial_bins = int(spatial_bins)
        self.channel_bins = int(channel_bins)
        self.population_bins = int(population_bins)
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        self.min_support = int(min_support)
        self.replace = bool(replace)
        self.seed = int(seed)
        self.buffers: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.luts: Dict[str, DensePrototypeLUT] = {}
        self.metrics: Dict[str, LUTRunMetrics] = defaultdict(LUTRunMetrics)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def finalize(self) -> None:
        for name, lut in self.luts.items():
            lut.finalize()
            self.metrics[name].update_entries(lut.supported_entries)

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {name: metric.summary() for name, metric in self.metrics.items()}

    def _register(self) -> None:
        for name, module in self.model.named_modules():
            if module.__class__.__name__ != "BasicBlock":
                continue
            if not all(hasattr(module, attr) for attr in ("conv1", "conv2", "sn1", "sn2")):
                continue
            self.handles.append(module.sn1.register_forward_hook(self._make_sn1_hook(name)))
            self.handles.append(module.sn2.register_forward_hook(self._make_sn2_hook(name)))

    def _make_sn1_hook(self, prefix: str):
        def hook(_module, _inputs, output):
            self.buffers[prefix]["sn1"] = output.detach()
            self.metrics[prefix].update_sparsity(output)
            return None

        return hook

    def _make_sn2_hook(self, prefix: str):
        def hook(_module, _inputs, output):
            return self._consume(prefix, output)

        return hook

    @torch.no_grad()
    def _consume(self, prefix: str, sn2_output: torch.Tensor) -> Optional[torch.Tensor]:
        if "sn1" not in self.buffers[prefix]:
            return None
        address, coarse_address, address_space, coarse_address_space = sew_local_conv_address(
            spike=self.buffers[prefix]["sn1"],
            out_channels=int(sn2_output.shape[2]),
            kernel_size=self.kernel_size,
            padding=self.padding,
            spatial_bins=self.spatial_bins,
            channel_bins=self.channel_bins,
            population_bins=self.population_bins,
        )
        if self.mode == "token_channel_mean":
            address = coarse_address
            address_space = coarse_address_space
        lut = self.luts.get(prefix)
        if lut is None:
            lut = DensePrototypeLUT(address_space, mode=self.mode, min_support=self.min_support, seed=self.seed)
            self.luts[prefix] = lut
        if not self.replace:
            lut.update(address, sn2_output)
            return None
        pred, seen = lut.predict(address)
        self.metrics[prefix].update_lookup(seen)
        return pred.reshape_as(sn2_output).to(sn2_output.dtype)
