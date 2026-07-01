from __future__ import annotations

from typing import Dict, Mapping, Tuple, TypeVar

import torch
from torch import nn

from .neuron import DenseLUTIFNeuron, QuantizedArithmeticLIF

ModuleT = TypeVar("ModuleT", bound=nn.Module)


def set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    if "." not in name:
        setattr(root, name, module)
        return
    parent_name, child_name = name.rsplit(".", 1)
    setattr(root.get_submodule(parent_name), child_name, module)


def _common(module: nn.Module) -> Dict[str, object]:
    return {
        "tau": float(getattr(module, "tau")),
        "decay_input": bool(getattr(module, "decay_input")),
        "v_threshold": float(getattr(module, "v_threshold")),
        "v_reset": getattr(module, "v_reset"),
    }


def _device_and_dtype(module: nn.Module, fallback: nn.Module) -> Tuple[torch.device, torch.dtype]:
    for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
        return tensor.device, tensor.dtype
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device, tensor.dtype
    for tensor in list(fallback.parameters()) + list(fallback.buffers()):
        if tensor.is_floating_point():
            return tensor.device, tensor.dtype
    return torch.device("cpu"), torch.float32


def _match_original_device(replacement: nn.Module, original: nn.Module, fallback: nn.Module) -> nn.Module:
    device, dtype = _device_and_dtype(original, fallback)
    return replacement.to(device=device, dtype=dtype)


def replace_with_dense_lutif(
    model: nn.Module,
    targets: Mapping[str, nn.Module],
    ranges: Mapping[str, Mapping[str, Tuple[float, float]]],
    *,
    state_bits: int,
    input_bits: int,
    surrogate_slope: float = 2.0,
    learn_threshold: bool = True,
) -> Dict[str, DenseLUTIFNeuron]:
    replacements: Dict[str, DenseLUTIFNeuron] = {}
    for name, original in list(targets.items()):
        replacement = DenseLUTIFNeuron(
            **_common(original),
            x_range=ranges[name]["x"],
            v_range=ranges[name]["v"],
            state_bits=state_bits,
            input_bits=input_bits,
            surrogate_slope=surrogate_slope,
            learn_threshold=learn_threshold,
        )
        replacement = _match_original_device(replacement, original, model)
        set_submodule(model, name, replacement)
        replacements[name] = replacement
    return replacements


def replace_with_quantized_arithmetic(
    model: nn.Module,
    targets: Mapping[str, nn.Module],
    ranges: Mapping[str, Mapping[str, Tuple[float, float]]],
    *,
    state_bits: int,
    input_bits: int,
    surrogate_slope: float = 2.0,
) -> Dict[str, QuantizedArithmeticLIF]:
    replacements: Dict[str, QuantizedArithmeticLIF] = {}
    for name, original in list(targets.items()):
        replacement = QuantizedArithmeticLIF(
            **_common(original),
            x_range=ranges[name]["x"],
            v_range=ranges[name]["v"],
            state_bits=state_bits,
            input_bits=input_bits,
            surrogate_slope=surrogate_slope,
        )
        replacement = _match_original_device(replacement, original, model)
        set_submodule(model, name, replacement)
        replacements[name] = replacement
    return replacements


def reset_metrics(modules: Mapping[str, nn.Module]) -> None:
    for module in modules.values():
        reset = getattr(module, "reset_metrics", None)
        if callable(reset):
            reset()
        else:
            for key in ("executions", "clip_count", "input_count", "quantization_sse"):
                if hasattr(module, key):
                    setattr(module, key, 0 if key != "quantization_sse" else 0.0)


def metadata_summary(modules: Mapping[str, nn.Module]) -> Dict[str, object]:
    per_module = {}
    total_bytes = 0
    total_entries = 0
    executed = 0
    clip_count = 0
    input_count = 0
    for name, module in modules.items():
        summary = module.summary() if hasattr(module, "summary") else {}
        per_module[name] = summary
        total_bytes += int(summary.get("metadata_bytes", 0))
        total_entries += int(summary.get("table_entries", 0))
        executed += int(float(summary.get("executions", 0)) > 0)
        clip_count += int(getattr(module, "clip_count", 0))
        input_count += int(getattr(module, "input_count", 0))
    return {
        "modules": len(modules),
        "modules_executed": executed,
        "table_entries": total_entries,
        "metadata_bytes": total_bytes,
        "metadata_kib": total_bytes / 1024.0,
        "aggregate_clip_rate": clip_count / input_count if input_count else 0.0,
        "per_module": per_module,
    }


def trainable_parameters(modules: Mapping[str, nn.Module]):
    for module in modules.values():
        yield from module.parameters()


def regularization_terms(modules: Mapping[str, DenseLUTIFNeuron]) -> Dict[str, torch.Tensor]:
    terms = [module.regularization() for module in modules.values()]
    if not terms:
        raise ValueError("no LUT-IF modules were provided")
    return {
        key: torch.stack([term[key] for term in terms]).mean()
        for key in terms[0]
    }
