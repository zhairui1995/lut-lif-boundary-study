from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, MutableMapping, Optional


def _percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * pct
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    alpha = pos - lo
    return float(sorted_values[lo] * (1.0 - alpha) + sorted_values[hi] * alpha)


@dataclass
class VarianceStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def update_many(self, values: Iterable[float]) -> None:
        for value in values:
            self.update(float(value))

    @property
    def variance(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.m2 / self.count

    def summary(self) -> Dict[str, float]:
        return {
            "count": int(self.count),
            "mean": float(self.mean),
            "variance": float(self.variance),
        }


@dataclass
class BucketStats:
    address_space: Optional[int] = None
    buckets: MutableMapping[int, VarianceStats] = field(default_factory=dict)

    def update_many(self, addresses: Iterable[int], responses: Iterable[float]) -> None:
        for address, response in zip(addresses, responses):
            key = int(address)
            bucket = self.buckets.get(key)
            if bucket is None:
                bucket = VarianceStats()
                self.buckets[key] = bucket
            bucket.update(float(response))

    def summary(self) -> Dict[str, object]:
        counts = sorted(bucket.count for bucket in self.buckets.values())
        total = sum(counts)
        unique = len(counts)
        if total <= 0 or unique <= 0:
            occupancy = {
                "min": 0.0,
                "p50": 0.0,
                "mean": 0.0,
                "p90": 0.0,
                "max": 0.0,
            }
            singleton_fraction = 0.0
            conditional_variance = 0.0
        else:
            occupancy = {
                "min": float(counts[0]),
                "p50": _percentile([float(x) for x in counts], 0.50),
                "mean": float(total / unique),
                "p90": _percentile([float(x) for x in counts], 0.90),
                "max": float(counts[-1]),
            }
            singleton_fraction = sum(1 for count in counts if count == 1) / unique
            conditional_variance = (
                sum(bucket.m2 for bucket in self.buckets.values()) / total
            )

        if self.address_space and self.address_space > 0:
            address_coverage = unique / float(self.address_space)
        else:
            address_coverage = None

        return {
            "num_samples": int(total),
            "unique_addresses": int(unique),
            "address_space": int(self.address_space) if self.address_space else None,
            "address_coverage": address_coverage,
            "bucket_occupancy": occupancy,
            "singleton_fraction": float(singleton_fraction),
            "conditional_variance": float(conditional_variance),
        }
