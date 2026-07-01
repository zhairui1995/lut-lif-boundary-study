"""Trainable lookup-table integrate-and-fire modules."""

from .neuron import DenseLUTIFNeuron, QuantizedArithmeticLIF
from .replace import (
    metadata_summary,
    replace_with_dense_lutif,
    replace_with_quantized_arithmetic,
)

__all__ = [
    "DenseLUTIFNeuron",
    "QuantizedArithmeticLIF",
    "metadata_summary",
    "replace_with_dense_lutif",
    "replace_with_quantized_arithmetic",
]
