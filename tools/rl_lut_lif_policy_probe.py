#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch

from lut_if.replace import metadata_summary, replace_with_quantized_arithmetic
from tools.cnl_lut_lif_e0 import collect_current_moments, replace_with_cnl_lut_lif
from tools.lut_if_poc import build_model, build_ranges, evaluate_pair, freeze, generic_module_summary, loader_config
from tools.qkformer_lut_current_unit_probe import reset_snn
from tools.qkformer_lut_e2_replace import build_loader
from tools.qkformer_native_affine_lut_probe import discover_lif_targets, replace_lif_targets


ACTIONS = ("posthoc", "quant", "cnl")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_name(target_name: str) -> str:
    parts = target_name.split(".")
    if not parts:
        return target_name
    for width in (3, 2):
        if len(parts) >= width and any(token.startswith(("stage", "block", "layer")) for token in parts[:width]):
            return ".".join(parts[:width])
    if parts[0] in {"patch_embed", "head", "classifier"}:
        return parts[0]
    if len(parts) >= 2 and parts[1].isdigit():
        return ".".join(parts[:2])
    return parts[0]


def select_targets(
    targets: Mapping[str, torch.nn.Module],
    groups: Mapping[str, str],
    action: str,
    policy: Mapping[str, str],
) -> Dict[str, torch.nn.Module]:
    return {name: module for name, module in targets.items() if policy[groups[name]] == action}


def reward(row: Mapping[str, float], *, logit_weight: float) -> float:
    return -float(row["drop_top1"]) - float(logit_weight) * float(row["logit_mse_per_sample"])


def module_summary_by_action(replacements: Mapping[str, Mapping[str, torch.nn.Module]]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for action, modules in replacements.items():
        if action == "posthoc":
            summary[action] = generic_module_summary(modules)
        else:
            summary[action] = metadata_summary(modules)
    return summary


def build_candidate(
    root: Path,
    args,
    device: torch.device,
    ranges: Mapping[str, Mapping[str, Tuple[float, float]]],
    moments: Mapping[str, Mapping[str, torch.Tensor]],
    policy: Mapping[str, str],
    groups: Mapping[str, str],
):
    model, checkpoint = build_model(root, args, device)
    freeze(model)
    targets = discover_lif_targets(model)
    replacements: Dict[str, Mapping[str, torch.nn.Module]] = {}

    posthoc_targets = select_targets(targets, groups, "posthoc", policy)
    quant_targets = select_targets(targets, groups, "quant", policy)
    cnl_targets = select_targets(targets, groups, "cnl", policy)

    if posthoc_targets:
        replacements["posthoc"] = replace_lif_targets(model, posthoc_targets, args.state_bits, dict(ranges))
    else:
        replacements["posthoc"] = {}
    if quant_targets:
        replacements["quant"] = replace_with_quantized_arithmetic(
            model,
            quant_targets,
            ranges,
            state_bits=args.state_bits,
            input_bits=args.input_bits,
        )
    else:
        replacements["quant"] = {}
    if cnl_targets:
        replacements["cnl"] = replace_with_cnl_lut_lif(
            model,
            cnl_targets,
            ranges,
            moments,
            state_bits=args.state_bits,
            input_bits=args.input_bits,
            normalize_current=True,
            time_steps=args.time_step,
        )
    else:
        replacements["cnl"] = {}
    flat = {}
    for action, modules in replacements.items():
        for name, module in modules.items():
            flat[f"{action}:{name}"] = module
    return model, checkpoint, replacements, flat


def evaluate_policy(
    *,
    root: Path,
    args,
    clean: torch.nn.Module,
    loader,
    device: torch.device,
    ranges,
    moments,
    groups,
    policy,
    tag: str,
    max_batches: Optional[int],
    logit_weight: float,
) -> Dict[str, object]:
    model, checkpoint, replacements, flat = build_candidate(root, args, device, ranges, moments, policy, groups)
    metrics = evaluate_pair(clean, model, loader, device, flat, max_batches)
    storage = module_summary_by_action(replacements)
    expected = len(policy_targets(groups, policy))
    executed = sum(int(getattr(module, "executions", 0) > 0) for module in flat.values())
    row = {
        "tag": tag,
        "checkpoint": checkpoint,
        "reward": reward(metrics, logit_weight=logit_weight),
        "expected_replaced_targets": expected,
        "executed_replaced_targets": executed,
        "all_replaced_targets_executed": executed == expected,
        **metrics,
        "storage": storage,
        "policy": dict(policy),
    }
    reset_snn(model)
    return row


def policy_targets(groups: Mapping[str, str], policy: Mapping[str, str]) -> Dict[str, str]:
    return {name: policy[group] for name, group in groups.items()}


def build_report(metrics: Mapping[str, object]) -> str:
    gate = metrics["gate"]
    final = metrics["final"]
    controls = metrics["controls"]
    search_controls = metrics["search_controls"]
    search_controls_table = "\n".join(
        f"| {row['tag']} | {row['candidate_top1']:.4f} | {row['drop_top1']:.4f} | "
        f"{row['logit_mse_per_sample']:.6f} | {row['reward']:.6f} |"
        for row in search_controls
    )
    controls_table = "\n".join(
        f"| {row['tag']} | {row['candidate_top1']:.4f} | {row['drop_top1']:.4f} | "
        f"{row['logit_mse_per_sample']:.6f} | {row['reward']:.6f} |"
        for row in controls
    )
    return f"""# RL-Guided QK-LUT-LIF Policy Probe

Status: **{gate['verdict']}**

## Scope

- QKFormer {metrics['protocol']['family']}, seed {metrics['protocol']['seed']}, T={metrics['protocol']['time_step']}
- All discovered MultiStepLIF targets grouped into {len(metrics['groups'])} policy groups
- Actions per group: `posthoc`, `quant`, `cnl`
- Fixed coordinate-bandit budget: {metrics['protocol']['policy_rounds']} round(s), no seed/checkpoint/bit-width search

## Search-Subset Fixed-Policy Controls

These rows use the same subset as the policy search and are not used for the
formal full-validation gate.

| Policy | Acc@1 | Drop | Logit MSE/sample | Reward |
|---|---:|---:|---:|---:|
{search_controls_table}

## Full-Evaluation Fixed-Policy Controls

| Policy | Acc@1 | Drop | Logit MSE/sample | Reward |
|---|---:|---:|---:|---:|
{controls_table}

## Learned Policy Result

- Acc@1: **{final['candidate_top1']:.4f}%**
- Drop: **{final['drop_top1']:.4f} pp**
- Logit MSE/sample: **{final['logit_mse_per_sample']:.6f}**
- Reward: **{final['reward']:.6f}**

## Gate

- Beats best fixed control by reward: **{gate['beats_best_fixed_control_reward']}**
- Drop is within threshold: **{gate['drop_within_threshold']}**
- All selected replacement modules executed: **{gate['all_selected_modules_executed']}**

## Claim Boundary

This run tests a restricted policy-search mechanism for selecting LUT-LIF
replacement variants inside QKFormer. It can motivate a QK-LUT-LIF Transformer
architecture only if the policy improves the matched controls under the fixed
budget. It does not establish broad transfer, near-SOTA accuracy, hardware
efficiency, or LL-ViT compatibility.
"""


def run(args) -> Dict[str, object]:
    set_seed(args.seed)
    root = Path(args.root).resolve()
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.local_cuda_index}" if torch.cuda.is_available() else "cpu")

    train_loader = build_loader(loader_config(args, "train", True), device)
    val_loader = build_loader(loader_config(args, "validation", False), device)

    clean, clean_checkpoint = build_model(root, args, device)
    freeze(clean)
    clean.eval()
    setattr(clean, "_cnl_time_steps", int(args.time_step))

    calibration_batches, ranges, range_adjustments = build_ranges(clean, train_loader, device, args)
    clean_targets = discover_lif_targets(clean)
    if not clean_targets:
        raise RuntimeError("no MultiStepLIFNode targets were discovered")
    groups = {name: group_name(name) for name in clean_targets}
    group_names = sorted(set(groups.values()))
    moment_batches, moments = collect_current_moments(
        clean,
        train_loader,
        device,
        clean_targets,
        ranges,
        input_bits=args.input_bits,
        batches=args.calib_batches,
    )

    search_controls = []
    for action in ACTIONS:
        policy = {group: action for group in group_names}
        search_controls.append(
            evaluate_policy(
                root=root,
                args=args,
                clean=clean,
                loader=val_loader,
                device=device,
                ranges=ranges,
                moments=moments,
                groups=groups,
                policy=policy,
                tag=f"all_{action}",
                max_batches=args.search_eval_batches,
                logit_weight=args.logit_reward_weight,
            )
        )

    current_policy = {group: "cnl" for group in group_names}
    search_rows = []
    for round_idx in range(1, args.policy_rounds + 1):
        for group in group_names:
            candidates = []
            for action in ACTIONS:
                trial = dict(current_policy)
                trial[group] = action
                row = evaluate_policy(
                    root=root,
                    args=args,
                    clean=clean,
                    loader=val_loader,
                    device=device,
                    ranges=ranges,
                    moments=moments,
                    groups=groups,
                    policy=trial,
                    tag=f"round{round_idx}:{group}:{action}",
                    max_batches=args.search_eval_batches,
                    logit_weight=args.logit_reward_weight,
                )
                candidates.append(row)
                search_rows.append(
                    {
                        "round": round_idx,
                        "group": group,
                        "action": action,
                        "reward": row["reward"],
                        "candidate_top1": row["candidate_top1"],
                        "drop_top1": row["drop_top1"],
                        "logit_mse_per_sample": row["logit_mse_per_sample"],
                    }
                )
            best = max(candidates, key=lambda item: float(item["reward"]))
            current_policy[group] = str(best["policy"][group])

    final = evaluate_policy(
        root=root,
        args=args,
        clean=clean,
        loader=val_loader,
        device=device,
        ranges=ranges,
        moments=moments,
        groups=groups,
        policy=current_policy,
        tag="learned_policy",
        max_batches=args.final_eval_batches,
        logit_weight=args.logit_reward_weight,
    )
    controls = []
    for action in ACTIONS:
        policy = {group: action for group in group_names}
        controls.append(
            evaluate_policy(
                root=root,
                args=args,
                clean=clean,
                loader=val_loader,
                device=device,
                ranges=ranges,
                moments=moments,
                groups=groups,
                policy=policy,
                tag=f"full_all_{action}",
                max_batches=args.final_eval_batches,
                logit_weight=args.logit_reward_weight,
            )
        )
    best_control = max(controls, key=lambda item: float(item["reward"]))
    beats_control = float(final["reward"]) > float(best_control["reward"])
    drop_within = float(final["drop_top1"]) <= float(args.max_drop)
    all_executed = bool(final["all_replaced_targets_executed"])
    smoke_mode = args.final_eval_batches is not None
    gate_pass = bool(beats_control and drop_within and all_executed)
    smoke_pass = bool(all_executed and math.isfinite(float(final["reward"])))

    policy_rows = [
        {
            "target": name,
            "group": groups[name],
            "action": current_policy[groups[name]],
        }
        for name in sorted(groups)
    ]
    summary_rows = []
    for row in [*search_controls, *controls, final]:
        summary_rows.append(
            {
                "tag": row["tag"],
                "candidate_top1": row["candidate_top1"],
                "drop_top1": row["drop_top1"],
                "logit_mse_per_sample": row["logit_mse_per_sample"],
                "reward": row["reward"],
                "all_replaced_targets_executed": row["all_replaced_targets_executed"],
            }
        )

    metrics = {
        "experiment": "rl_guided_qklut_lif_policy_probe",
        "status": "completed",
        "protocol": {
            "family": args.family,
            "seed": args.seed,
            "time_step": args.time_step,
            "state_bits": args.state_bits,
            "input_bits": args.input_bits,
            "calibration_batches": calibration_batches,
            "moment_batches": moment_batches,
            "search_eval_batches": args.search_eval_batches,
            "final_eval_batches": args.final_eval_batches,
            "policy_rounds": args.policy_rounds,
            "max_drop": args.max_drop,
            "logit_reward_weight": args.logit_reward_weight,
            "checkpoint": args.checkpoint,
            "clean_checkpoint_load": clean_checkpoint,
            "data_dir": args.data_dir,
            "range_adjustments": range_adjustments,
        },
        "groups": groups,
        "search_controls": search_controls,
        "controls": controls,
        "search_rows": search_rows,
        "final_policy": current_policy,
        "final_policy_by_target": policy_rows,
        "final": final,
        "gate": {
            "smoke_mode": smoke_mode,
            "beats_best_fixed_control_reward": beats_control,
            "best_fixed_control": best_control["tag"],
            "drop_within_threshold": drop_within,
            "all_selected_modules_executed": all_executed,
            "pass": smoke_pass if smoke_mode else gate_pass,
            "formal_full_gate_pass": gate_pass,
            "verdict": ("SMOKE-PASS" if smoke_pass else "SMOKE-FAIL") if smoke_mode else ("GO" if gate_pass else "NO-GO"),
        },
    }
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, default=str)
    save_csv(result_dir / "summary.csv", summary_rows)
    save_csv(result_dir / "policy_by_target.csv", policy_rows)
    save_csv(result_dir / "policy_search.csv", search_rows)
    (result_dir / "report.md").write_text(build_report(metrics), encoding="utf-8")
    print(json.dumps({"result_dir": str(result_dir), "verdict": metrics["gate"]["verdict"]}, sort_keys=True))
    return metrics


def none_if_negative(value: Optional[int]) -> Optional[int]:
    if value is None or value < 0:
        return None
    return value


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Restricted RL/bandit probe for QK-LUT-LIF insertion policies")
    p.add_argument("--root", default=".")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--result-dir", required=True)
    p.add_argument("--family", choices=["cifar10", "cifar100"], default="cifar100")
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--local-cuda-index", type=int, default=0)
    p.add_argument("--time-step", type=int, default=4)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--calib-batches", type=int, default=128)
    p.add_argument("--range-margin", type=float, default=0.05)
    p.add_argument("--state-bits", type=int, default=6)
    p.add_argument("--input-bits", type=int, default=6)
    p.add_argument("--search-eval-batches", type=int, default=8)
    p.add_argument("--final-eval-batches", type=int, default=-1)
    p.add_argument("--policy-rounds", type=int, default=1)
    p.add_argument("--max-drop", type=float, default=1.50)
    p.add_argument("--logit-reward-weight", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    parsed = parser().parse_args()
    parsed.search_eval_batches = none_if_negative(parsed.search_eval_batches)
    parsed.final_eval_batches = none_if_negative(parsed.final_eval_batches)
    run(parsed)
