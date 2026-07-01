from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch


@dataclass
class LUTAudit:
    calls: Dict[str, int] = field(default_factory=dict)
    table_entries: Dict[str, int] = field(default_factory=dict)

    def record(self, name: str, entries: int) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1
        self.table_entries[name] = max(self.table_entries.get(name, 0), int(entries))

    def summary(self) -> Dict[str, object]:
        return {
            "calls": dict(sorted(self.calls.items())),
            "table_entries": dict(sorted(self.table_entries.items())),
            "unique_table_entries": int(sum(self.table_entries.values())),
        }


class ExactIntegerLUT:
    """Exact lookup primitives for bounded non-negative integer tensors."""

    def __init__(self, audit: LUTAudit) -> None:
        self.audit = audit
        self._cache: Dict[Tuple[str, int, int, str], torch.Tensor] = {}

    def _table(
        self,
        kind: str,
        max_a: int,
        max_b: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = (kind, int(max_a), int(max_b), str(device))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        a = torch.arange(max_a + 1, device=device, dtype=torch.long).view(-1, 1)
        b = torch.arange(max_b + 1, device=device, dtype=torch.long).view(1, -1)
        if kind == "add":
            table = a + b
        elif kind == "max":
            table = torch.maximum(a, b)
        elif kind == "and":
            table = torch.bitwise_and(a, b)
        else:
            raise ValueError(f"unsupported table kind: {kind}")
        table = table.reshape(-1)
        self._cache[key] = table
        return table

    def add(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        max_a: int,
        max_b: int,
    ) -> torch.Tensor:
        if a.shape != b.shape:
            a, b = torch.broadcast_tensors(a, b)
        table = self._table("add", max_a, max_b, a.device)
        self.audit.record(f"add_{max_a}x{max_b}", table.numel())
        return table[a.long() * (max_b + 1) + b.long()]

    def binary_and(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape != b.shape:
            a, b = torch.broadcast_tensors(a, b)
        table = self._table("and", 1, 1, a.device)
        self.audit.record("binary_and", table.numel())
        index = torch.bitwise_or(torch.bitwise_left_shift(a.long(), 1), b.long())
        return table[index]

    def binary_select(
        self,
        selector: torch.Tensor,
        value: torch.Tensor,
        value_max: int,
    ) -> torch.Tensor:
        if selector.shape != value.shape:
            selector, value = torch.broadcast_tensors(selector, value)
        key = ("select", 1, int(value_max), str(value.device))
        table = self._cache.get(key)
        if table is None:
            values = torch.arange(
                value_max + 1,
                device=value.device,
                dtype=torch.long,
            )
            table = torch.cat((torch.zeros_like(values), values), dim=0)
            self._cache[key] = table
        self.audit.record(f"binary_select_{value_max}", table.numel())
        return table[selector.long() * (value_max + 1) + value.long()]

    def reduce_sum(
        self,
        value: torch.Tensor,
        dim: int,
        element_max: int,
    ) -> Tuple[torch.Tensor, int]:
        moved = value.movedim(dim, -1).long()
        bound = int(element_max)
        while moved.shape[-1] > 1:
            if moved.shape[-1] % 2:
                moved = torch.cat(
                    (moved, torch.zeros_like(moved[..., :1])),
                    dim=-1,
                )
            moved = self.add(
                moved[..., 0::2],
                moved[..., 1::2],
                bound,
                bound,
            )
            bound *= 2
        return moved.squeeze(-1), bound

    def reduce_max(
        self,
        value: torch.Tensor,
        dim: int,
        value_max: int,
    ) -> torch.Tensor:
        moved = value.movedim(dim, -1).long()
        bound = int(value_max)
        while moved.shape[-1] > 1:
            if moved.shape[-1] % 2:
                moved = torch.cat((moved, moved[..., -1:]), dim=-1)
            left = moved[..., 0::2]
            right = moved[..., 1::2]
            table = self._table("max", bound, bound, moved.device)
            self.audit.record(f"max_{bound}x{bound}", table.numel())
            moved = table[left * (bound + 1) + right]
        return moved.squeeze(-1)

    def binary_popcount(
        self,
        value: torch.Tensor,
        dim: int,
    ) -> Tuple[torch.Tensor, int]:
        moved = value.movedim(dim, -1).long()
        group_bits = 8
        pad = (-moved.shape[-1]) % group_bits
        if pad:
            moved = torch.cat(
                (
                    moved,
                    torch.zeros(
                        (*moved.shape[:-1], pad),
                        device=moved.device,
                        dtype=moved.dtype,
                    ),
                ),
                dim=-1,
            )
        grouped = moved.reshape(*moved.shape[:-1], -1, group_bits)
        packed = torch.zeros(
            grouped.shape[:-1],
            device=moved.device,
            dtype=torch.long,
        )
        for bit in range(group_bits):
            packed = torch.bitwise_or(
                packed,
                torch.bitwise_left_shift(grouped[..., bit], bit),
            )
        key = ("popcount8", 255, 0, str(moved.device))
        table = self._cache.get(key)
        if table is None:
            table = torch.tensor(
                [bin(index).count("1") for index in range(256)],
                device=moved.device,
                dtype=torch.long,
            )
            self._cache[key] = table
        self.audit.record("popcount8", table.numel())
        reduced, bound = self.reduce_sum(table[packed], -1, group_bits)
        return reduced, min(bound, moved.shape[-1])

    def scaled_value(
        self,
        value: torch.Tensor,
        value_max: int,
        scale: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (f"scale_{scale}", int(value_max), 0, str(value.device))
        table = self._cache.get(key)
        if table is None:
            table = (
                torch.arange(value_max + 1, device=value.device, dtype=dtype)
                * float(scale)
            )
            self._cache[key] = table
        self.audit.record(f"scale_{value_max}_{scale}", table.numel())
        return table[value.long()]


def token_qk_interaction(
    q: torch.Tensor,
    k: torch.Tensor,
    ops: ExactIntegerLUT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_population, _ = ops.binary_popcount(q, dim=3)
    return q_population.unsqueeze(3), k


def ssa_interaction(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ops: ExactIntegerLUT,
    output_chunk: int = 8,
) -> torch.Tensor:
    """Exact table-addressed binary SSA arithmetic for [T, B, H, N, D]."""

    time_steps, batch, heads, tokens, depth = q.shape
    outputs = []
    k_by_depth = k.transpose(-2, -1)
    for start in range(0, depth, output_chunk):
        end = min(depth, start + output_chunk)
        products = ops.binary_and(
            k_by_depth.unsqueeze(-1),
            v[..., start:end].unsqueeze(-3),
        )
        ktv, _ = ops.binary_popcount(products, dim=-2)
        selected = ops.binary_select(
            q.unsqueeze(-1),
            ktv.unsqueeze(-3),
            value_max=tokens,
        )
        qktv, qktv_bound = ops.reduce_sum(
            selected,
            dim=-2,
            element_max=tokens,
        )
        exact_max = tokens * depth
        if qktv_bound < exact_max:
            raise RuntimeError(
                f"invalid SSA bound {qktv_bound} < {exact_max}"
            )
        outputs.append(
            ops.scaled_value(qktv, exact_max, 0.125, q.dtype)
        )
    return torch.cat(outputs, dim=-1).reshape(
        time_steps,
        batch,
        heads,
        tokens,
        depth,
    )
