from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn


class _FastSigmoidSpike(torch.autograd.Function):
    """Hard spike in the forward pass with a bounded surrogate gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.slope = float(slope)
        return (x >= 0).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        slope = ctx.slope
        grad = slope / (1.0 + slope * x.abs()).pow(2)
        return grad_output * grad, None


def spike_function(x: torch.Tensor, slope: float = 2.0) -> torch.Tensor:
    return _FastSigmoidSpike.apply(x, slope)


def _safe_span(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return (hi - lo).clamp_min(torch.finfo(lo.dtype).eps)


def _index_coordinates(
    value: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    levels: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    clipped = value.clamp(min=float(lo.item()), max=float(hi.item()))
    position = (clipped - lo) / _safe_span(lo, hi) * float(levels - 1)
    lower = position.floor().long().clamp_(0, levels - 1)
    upper = (lower + 1).clamp_(0, levels - 1)
    weight = position - lower.to(dtype=position.dtype)
    clip_mask = (value < lo) | (value > hi)
    return lower, upper, weight, clip_mask


def _bilinear_lookup(
    table: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    v_range: Tuple[torch.Tensor, torch.Tensor],
    x_range: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    v0, v1, vw, v_clip = _index_coordinates(v, v_range[0], v_range[1], table.shape[0])
    x0, x1, xw, x_clip = _index_coordinates(x, x_range[0], x_range[1], table.shape[1])
    t00 = table[v0, x0]
    t01 = table[v0, x1]
    t10 = table[v1, x0]
    t11 = table[v1, x1]
    top = t00 + (t01 - t00) * xw
    bottom = t10 + (t11 - t10) * xw
    output = top + (bottom - top) * vw
    return output, v_clip | x_clip


def _nearest_lookup(
    table: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    v_range: Tuple[torch.Tensor, torch.Tensor],
    x_range: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    v0, v1, vw, v_clip = _index_coordinates(v, v_range[0], v_range[1], table.shape[0])
    x0, x1, xw, x_clip = _index_coordinates(x, x_range[0], x_range[1], table.shape[1])
    vi = torch.where(vw >= 0.5, v1, v0)
    xi = torch.where(xw >= 0.5, x1, x0)
    return table[vi, xi], v_clip | x_clip


class DenseLUTIFNeuron(nn.Module):
    """Trainable dense state/input transition lookup.

    The table stores the membrane value immediately before thresholding. During
    training, bilinear interpolation supplies gradients to the table and its
    inputs. Evaluation uses a deterministic nearest-neighbour address. The
    spike and reset operations retain the IF/LIF inductive bias.
    """

    def __init__(
        self,
        *,
        tau: float,
        decay_input: bool,
        v_threshold: float,
        v_reset: float | None,
        x_range: Tuple[float, float],
        v_range: Tuple[float, float],
        state_bits: int = 6,
        input_bits: int = 6,
        surrogate_slope: float = 2.0,
        learn_threshold: bool = True,
        hard_eval: bool = True,
    ) -> None:
        super().__init__()
        if state_bits < 2 or input_bits < 2:
            raise ValueError("state_bits and input_bits must both be >= 2")
        if tau <= 1.0:
            raise ValueError("tau must be > 1 for the LIF initialization used here")
        self.tau = float(tau)
        self.decay_input = bool(decay_input)
        self.v_reset = 0.0 if v_reset is None else float(v_reset)
        self.state_bits = int(state_bits)
        self.input_bits = int(input_bits)
        self.state_levels = 1 << self.state_bits
        self.input_levels = 1 << self.input_bits
        self.surrogate_slope = float(surrogate_slope)
        self.hard_eval = bool(hard_eval)

        x_lo, x_hi = map(float, x_range)
        v_lo, v_hi = map(float, v_range)
        if not math.isfinite(x_lo + x_hi + v_lo + v_hi):
            raise ValueError("transition ranges must be finite")
        if x_hi <= x_lo or v_hi <= v_lo:
            raise ValueError("transition ranges must have positive width")

        self.register_buffer("x_lo", torch.tensor(x_lo, dtype=torch.float32))
        self.register_buffer("x_hi", torch.tensor(x_hi, dtype=torch.float32))
        self.register_buffer("v_lo", torch.tensor(v_lo, dtype=torch.float32))
        self.register_buffer("v_hi", torch.tensor(v_hi, dtype=torch.float32))

        v_grid = torch.linspace(v_lo, v_hi, self.state_levels, dtype=torch.float32)
        x_grid = torch.linspace(x_lo, x_hi, self.input_levels, dtype=torch.float32)
        vv, xx = torch.meshgrid(v_grid, x_grid, indexing="ij")
        if self.decay_input:
            initialized = vv + (xx - vv) / self.tau
        else:
            initialized = vv * (1.0 - 1.0 / self.tau) + xx
        self.register_buffer("initial_table", initialized.clone())
        self.transition_table = nn.Parameter(initialized.clone())

        threshold = torch.tensor(float(v_threshold), dtype=torch.float32)
        if learn_threshold:
            self.threshold = nn.Parameter(threshold)
        else:
            self.register_buffer("threshold", threshold)

        self.v: torch.Tensor | float = 0.0
        self.v_seq: torch.Tensor | None = None
        self.executions = 0
        self.clip_count = 0
        self.input_count = 0
        self.quantization_sse = 0.0

    def reset(self) -> None:
        self.v = 0.0
        self.v_seq = None

    def reset_metrics(self) -> None:
        self.executions = 0
        self.clip_count = 0
        self.input_count = 0
        self.quantization_sse = 0.0

    def _state_tensor(self, reference: torch.Tensor) -> torch.Tensor:
        if isinstance(self.v, torch.Tensor):
            return self.v.to(device=reference.device, dtype=reference.dtype)
        return torch.full_like(reference, float(self.v))

    def _lookup(self, v: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        table = self.transition_table.to(device=x.device, dtype=x.dtype)
        v_range = (
            self.v_lo.to(device=x.device, dtype=x.dtype),
            self.v_hi.to(device=x.device, dtype=x.dtype),
        )
        x_range = (
            self.x_lo.to(device=x.device, dtype=x.dtype),
            self.x_hi.to(device=x.device, dtype=x.dtype),
        )
        if self.training or not self.hard_eval:
            return _bilinear_lookup(table, v, x, v_range, x_range)
        return _nearest_lookup(table, v, x, v_range, x_range)

    def _record_quantization_error(self, v: torch.Tensor, x: torch.Tensor) -> None:
        if self.training:
            return
        with torch.no_grad():
            table = self.transition_table.detach().to(dtype=x.dtype)
            table = table.to(device=x.device)
            v_range = (
                self.v_lo.to(device=x.device, dtype=x.dtype),
                self.v_hi.to(device=x.device, dtype=x.dtype),
            )
            x_range = (
                self.x_lo.to(device=x.device, dtype=x.dtype),
                self.x_hi.to(device=x.device, dtype=x.dtype),
            )
            hard, _ = _nearest_lookup(table, v, x, v_range, x_range)
            soft, _ = _bilinear_lookup(table, v, x, v_range, x_range)
            self.quantization_sse += float((hard - soft).float().square().sum().item())

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self._state_tensor(x)
        membrane, clip_mask = self._lookup(v, x)
        self._record_quantization_error(v, x)
        spike = spike_function(
            membrane - self.threshold.to(device=x.device, dtype=x.dtype),
            self.surrogate_slope,
        )
        next_v = membrane * (1.0 - spike) + self.v_reset * spike
        self.v = next_v
        if not self.training:
            self.clip_count += int(clip_mask.sum().item())
            self.input_count += int(x.numel())
        return spike

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim < 2:
            raise ValueError("DenseLUTIFNeuron expects a time-major tensor")
        self.executions += 1
        spikes = []
        states = []
        for step in range(x_seq.shape[0]):
            spikes.append(self.single_step_forward(x_seq[step]))
            states.append(self._state_tensor(x_seq[step]))
        self.v_seq = torch.stack(states, dim=0)
        return torch.stack(spikes, dim=0)

    def regularization(self) -> Dict[str, torch.Tensor]:
        delta = self.transition_table - self.initial_table
        smooth_v = (delta[1:, :] - delta[:-1, :]).square().mean()
        smooth_x = (delta[:, 1:] - delta[:, :-1]).square().mean()
        return {
            "delta_l2": delta.square().mean(),
            "neighbor_smoothness": smooth_v + smooth_x,
        }

    def table_entries(self) -> int:
        return int(self.transition_table.numel())

    def metadata_bytes(self) -> int:
        table_bytes = self.transition_table.numel() * self.transition_table.element_size()
        scalar_metadata = 7 * 4  # ranges, threshold, reset, tau
        return int(table_bytes + scalar_metadata)

    def summary(self) -> Dict[str, float]:
        values = max(self.input_count, 1)
        return {
            "executions": float(self.executions),
            "clip_rate": float(self.clip_count) / float(values),
            "hard_soft_mse": float(self.quantization_sse) / float(values),
            "table_entries": float(self.table_entries()),
            "metadata_bytes": float(self.metadata_bytes()),
            "threshold": float(self.threshold.detach().item()),
            "table_delta_rms": float(
                (self.transition_table.detach() - self.initial_table).square().mean().sqrt().item()
            ),
        }


class QuantizedArithmeticLIF(nn.Module):
    """Quantized-state arithmetic LIF baseline at matched state/input precision."""

    def __init__(
        self,
        *,
        tau: float,
        decay_input: bool,
        v_threshold: float,
        v_reset: float | None,
        x_range: Tuple[float, float],
        v_range: Tuple[float, float],
        state_bits: int = 6,
        input_bits: int = 6,
        surrogate_slope: float = 2.0,
    ) -> None:
        super().__init__()
        self.tau = float(tau)
        self.decay_input = bool(decay_input)
        self.v_reset = 0.0 if v_reset is None else float(v_reset)
        self.state_bits = int(state_bits)
        self.input_bits = int(input_bits)
        self.state_levels = 1 << self.state_bits
        self.input_levels = 1 << self.input_bits
        self.surrogate_slope = float(surrogate_slope)
        self.register_buffer("threshold", torch.tensor(float(v_threshold), dtype=torch.float32))
        self.register_buffer("x_lo", torch.tensor(float(x_range[0]), dtype=torch.float32))
        self.register_buffer("x_hi", torch.tensor(float(x_range[1]), dtype=torch.float32))
        self.register_buffer("v_lo", torch.tensor(float(v_range[0]), dtype=torch.float32))
        self.register_buffer("v_hi", torch.tensor(float(v_range[1]), dtype=torch.float32))
        self.v: torch.Tensor | float = 0.0
        self.v_seq: torch.Tensor | None = None
        self.executions = 0
        self.clip_count = 0
        self.input_count = 0

    def reset(self) -> None:
        self.v = 0.0
        self.v_seq = None

    @staticmethod
    def _quantize(value: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, levels: int):
        clipped = value.clamp(min=float(lo.item()), max=float(hi.item()))
        scale = float(levels - 1) / _safe_span(lo, hi)
        index = ((clipped - lo) * scale).round()
        decoded = index / scale + lo
        return decoded, (value < lo) | (value > hi)

    def _state_tensor(self, reference: torch.Tensor) -> torch.Tensor:
        if isinstance(self.v, torch.Tensor):
            return self.v.to(device=reference.device, dtype=reference.dtype)
        return torch.full_like(reference, float(self.v))

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self._state_tensor(x)
        x_q, x_clip = self._quantize(
            x,
            self.x_lo.to(device=x.device, dtype=x.dtype),
            self.x_hi.to(device=x.device, dtype=x.dtype),
            self.input_levels,
        )
        v_q, v_clip = self._quantize(
            v,
            self.v_lo.to(device=x.device, dtype=x.dtype),
            self.v_hi.to(device=x.device, dtype=x.dtype),
            self.state_levels,
        )
        if self.decay_input:
            membrane = v_q + (x_q - v_q) / self.tau
        else:
            membrane = v_q * (1.0 - 1.0 / self.tau) + x_q
        spike = spike_function(
            membrane - self.threshold.to(device=x.device, dtype=x.dtype),
            self.surrogate_slope,
        )
        self.v = membrane * (1.0 - spike) + self.v_reset * spike
        if not self.training:
            self.clip_count += int((x_clip | v_clip).sum().item())
            self.input_count += int(x.numel())
        return spike

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        self.executions += 1
        spikes = []
        states = []
        for step in range(x_seq.shape[0]):
            spikes.append(self.single_step_forward(x_seq[step]))
            states.append(self._state_tensor(x_seq[step]))
        self.v_seq = torch.stack(states, dim=0)
        return torch.stack(spikes, dim=0)

    def metadata_bytes(self) -> int:
        return 7 * 4

    def summary(self) -> Dict[str, float]:
        values = max(self.input_count, 1)
        return {
            "executions": float(self.executions),
            "clip_rate": float(self.clip_count) / float(values),
            "metadata_bytes": float(self.metadata_bytes()),
        }
