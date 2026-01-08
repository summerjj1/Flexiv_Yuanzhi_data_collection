#!/usr/bin/env python3
"""
Weight comparison script for JAX and PyTorch converted weights.
Compares two pickle files containing model weights to verify alignment.
"""

import argparse
import math
import pickle
import sys
from typing import Any, Dict, Tuple

import numpy as np
import torch


def generate_detailed_analysis_markdown(
    tensor1: torch.Tensor, tensor2: torch.Tensor, key: str
) -> str:
    """
    Generate detailed markdown analysis for a mismatched weight tensor.
    Returns markdown formatted string.
    """
    lines = []
    lines.append(f"## {key}")
    lines.append(f"**Shape:** {tensor1.shape}")
    lines.append("")

    # Convert to float32 for analysis
    t1 = tensor1.float()  # JAX weights
    t2 = tensor2.float()  # PyTorch weights

    # Calculate differences
    diff = t1 - t2
    abs_diff = torch.abs(diff)

    # 只在绝对误差 >= 1e-5 时计算相对误差（参考 debug_report.py）
    ref_diff = torch.where(
        abs_diff >= 1e-5,
        abs_diff * 100 / (torch.abs(t1) + 1e-6),
        torch.zeros_like(abs_diff),
    )

    # 计算有效的相对误差最大值
    valid_ref_diff = ref_diff[abs_diff >= 1e-5]
    if len(valid_ref_diff) > 0:
        lines.append(
            f"**Global Reference Difference:** {valid_ref_diff.max().item():.2e}%"
        )
    else:
        lines.append(
            "**Global Reference Difference:** N/A (all abs diff < 1e-5)"
        )

    # 找到最大误差
    max_diff = abs_diff.max().item()
    lines.append(f"**Maximum Absolute Difference:** {max_diff:.2e}")
    lines.append("")

    # Per-layer analysis (similar to debug_report.py)
    if len(tensor1.shape) > 1:
        lines.append("### Per-Layer Analysis")
        lines.append("")

        lines.append("#### Absolute Difference Details")
        lines.append(
            "| Layer | Max Abs Diff | Location | JAX Value | PyTorch Value | Difference |"
        )
        lines.append(
            "|-------|--------------|----------|-----------|---------------|------------|"
        )

        for layer_idx in range(tensor1.shape[0]):
            layer_diff = abs_diff[layer_idx].max().item()

            # 找到最大绝对误差位置
            layer_abs_diff = abs_diff[layer_idx]
            max_diff_idx = torch.argmax(layer_abs_diff.view(-1)).item()
            max_diff_indices = np.unravel_index(
                max_diff_idx, layer_abs_diff.shape
            )
            abs_location = (
                f"({layer_idx}, {', '.join(map(str, max_diff_indices))})"
            )

            # 获取该位置的值
            global_max_diff_indices = (layer_idx,) + max_diff_indices
            jax_val = t1[global_max_diff_indices].item()
            torch_val = t2[global_max_diff_indices].item()
            abs_diff_val = diff[global_max_diff_indices].item()

            lines.append(
                f"| {layer_idx} | {layer_diff:.2e} | {abs_location} | {jax_val:.6f} | {torch_val:.6f} | {abs_diff_val:.6e} |"
            )

        lines.append("")
        lines.append("#### Relative Difference Details")
        lines.append(
            "| Layer | Max Rel Diff | Location | JAX Value | PyTorch Value | Abs Diff | Rel Diff |"
        )
        lines.append(
            "|-------|--------------|----------|-----------|---------------|----------|----------|"
        )

        for layer_idx in range(tensor1.shape[0]):
            # 计算当前层的相对误差最大值
            layer_abs_diff = abs_diff[layer_idx]
            layer_ref_diff_tensor = ref_diff[layer_idx]
            valid_layer_ref_diff = layer_ref_diff_tensor[
                layer_abs_diff >= 1e-5
            ]
            layer_ref_diff_max = (
                valid_layer_ref_diff.max().item()
                if len(valid_layer_ref_diff) > 0
                else 0.0
            )

            # 找到最大相对误差位置
            valid_mask = layer_abs_diff >= 1e-5
            if valid_mask.any() and layer_ref_diff_max > 0:
                valid_ref_diff_flat = layer_ref_diff_tensor.view(-1)[
                    valid_mask.view(-1)
                ]
                max_valid_ref_idx = torch.argmax(valid_ref_diff_flat).item()
                valid_indices = torch.nonzero(
                    valid_mask.view(-1), as_tuple=True
                )[0]
                max_ref_diff_idx = valid_indices[max_valid_ref_idx].item()
                max_ref_diff_indices = np.unravel_index(
                    max_ref_diff_idx, layer_ref_diff_tensor.shape
                )
                global_max_ref_diff_indices = (
                    layer_idx,
                ) + max_ref_diff_indices
                rel_location = f"({layer_idx}, {', '.join(map(str, max_ref_diff_indices))})"

                # 获取该位置的值
                jax_val_ref = t1[global_max_ref_diff_indices].item()
                torch_val_ref = t2[global_max_ref_diff_indices].item()
                abs_diff_val_ref = diff[global_max_ref_diff_indices].item()
                rel_diff_val = ref_diff[global_max_ref_diff_indices].item()

                lines.append(
                    f"| {layer_idx} | {layer_ref_diff_max:.2e}% | {rel_location} | {jax_val_ref:.6f} | {torch_val_ref:.6f} | {abs_diff_val_ref:.6e} | {rel_diff_val:.2e}% |"
                )
            else:
                lines.append(
                    f"| {layer_idx} | N/A | N/A | N/A | N/A | N/A | N/A |"
                )

        lines.append("")

        # Find worst layer
        layer_max_diffs = [
            abs_diff[layer_idx].max().item()
            for layer_idx in range(abs_diff.shape[0])
        ]
        worst_layer = torch.argmax(torch.tensor(layer_max_diffs)).item()
        lines.append(
            f"**Worst Layer:** {worst_layer} (max diff: {layer_max_diffs[worst_layer]:.2e})"
        )
    else:
        lines.append(
            "**Note:** Tensor has only one dimension, per-layer analysis skipped."
        )
        lines.append("")

    return "\n".join(lines)


def generate_comparison_report(
    results: Dict[str, Dict[str, Any]],
    all_aligned: bool,
    jax_weights: Dict[str, torch.Tensor],
    torch_weights: Dict[str, torch.Tensor],
    jax_path: str,
    torch_path: str,
) -> str:
    """
    Generate a comprehensive markdown report of the weight comparison.
    """
    import datetime

    lines = []
    lines.append("# Weight Comparison Report")
    lines.append("")
    lines.append(
        f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append(f"**JAX Weights:** {jax_path}")
    lines.append(f"**PyTorch Weights:** {torch_path}")
    lines.append("")

    total_keys = len(results)
    shape_mismatches = sum(1 for r in results.values() if not r["shape_match"])
    value_mismatches = sum(
        1
        for r in results.values()
        if r["shape_match"] and not r["value_match"]
    )

    lines.append("## Summary")
    lines.append(f"- **Total Keys Compared:** {total_keys}")
    lines.append(f"- **Shape Mismatches:** {shape_mismatches}")
    lines.append(f"- **Value Mismatches:** {value_mismatches}")
    lines.append(
        f"- **Perfect Matches:** {total_keys - shape_mismatches - value_mismatches}"
    )

    if all_aligned:
        lines.append("- **Overall Status:** ✅ ALL WEIGHTS ALIGNED")
    else:
        lines.append("- **Overall Status:** ❌ WEIGHTS NOT FULLY ALIGNED")
    lines.append("")

    # Shape mismatches
    if shape_mismatches > 0:
        lines.append("## Shape Mismatches")
        for key, result in results.items():
            if not result["shape_match"]:
                lines.append(f"- **{key}:** Shape mismatch")
        lines.append("")

    # Value mismatches with detailed analysis
    if value_mismatches > 0:
        lines.append("## Value Mismatches")

        # Sort by max_abs_diff descending
        value_problems = [
            (k, r)
            for k, r in results.items()
            if r["shape_match"] and not r["value_match"]
        ]
        value_problems.sort(
            key=lambda x: x[1]["stats"]["max_abs_diff"], reverse=True
        )

        lines.append("### Overview")
        lines.append("| Weight Key | Max Abs Diff | Max Rel Diff | Shape |")
        lines.append("|------------|--------------|--------------|-------|")

        for key, result in value_problems:
            stats = result["stats"]
            shape_str = " × ".join(map(str, result["shape"]))
            lines.append(
                f"| {key} | {stats['max_abs_diff']:.2e} | {stats['max_rel_diff']:.2f}% | {shape_str} |"
            )

        lines.append("")
        lines.append("### Global Statistics for Mismatched Weights")
        lines.append("#### Absolute Difference Percentiles")
        all_abs_diffs = []
        all_rel_diffs = []

        for _, result in value_problems:
            stats = result["stats"]
            for p in [50, 75, 90, 95, 99, 99.9]:
                all_abs_diffs.append(stats["abs_percentiles"][f"p{p}"])
                all_rel_diffs.append(stats["rel_percentiles"][f"p{p}"])

        if all_abs_diffs:
            all_abs_diffs.sort()
            all_rel_diffs.sort()
            total_samples = len(all_abs_diffs)

            lines.append("| Percentile | Absolute Diff | Relative Diff |")
            lines.append("|------------|---------------|---------------|")
            for p in [50, 75, 90, 95, 99, 99.9]:
                idx = int(total_samples * p / 100)
                idx = min(idx, total_samples - 1)
                lines.append(
                    f"| P{p:5.1f}% | {all_abs_diffs[idx]:.2e} | {all_rel_diffs[idx]:.2f}% |"
                )

        lines.append("")
        lines.append("### Detailed Analysis")

        # Generate detailed analysis for each mismatched weight
        for key, result in value_problems:
            jax_tensor = jax_weights[key]
            torch_tensor = torch_weights[key]
            lines.append(
                generate_detailed_analysis_markdown(
                    jax_tensor, torch_tensor, key
                )
            )
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def print_detailed_layer_analysis(
    tensor1: torch.Tensor, tensor2: torch.Tensor, key: str
):
    """
    Print detailed per-layer analysis when weights don't match.
    Similar to check_diff function in debug_report.py
    """
    print(f"\n🔍 Detailed analysis for {key} (shape: {tensor1.shape})")
    print("=" * 80)

    # Convert to float32 for analysis
    t1 = tensor1.float()  # JAX weights
    t2 = tensor2.float()  # PyTorch weights

    # Calculate differences
    diff = t1 - t2
    abs_diff = torch.abs(diff)

    # 只在绝对误差 >= 1e-5 时计算相对误差（参考 debug_report.py）
    ref_diff = torch.where(
        abs_diff >= 1e-5,
        abs_diff * 100 / (torch.abs(t1) + 1e-6),
        torch.zeros_like(abs_diff),
    )

    # 计算有效的相对误差最大值（排除为0的部分）
    valid_ref_diff = ref_diff[abs_diff >= 1e-5]
    if len(valid_ref_diff) > 0:
        print(
            f"{key} Reference difference: {valid_ref_diff.max().item():.2e} %"
        )
    else:
        print(f"{key} Reference difference: N/A (all abs diff < 1e-5)")

    # 找到最大误差
    max_diff = abs_diff.max().item()
    print(f"Maximum absolute difference: {max_diff:.2e}")

    # 检查每层的最大误差（假设第一维度是层）
    if len(tensor1.shape) > 1:
        print("\nPer-layer maximum absolute differences:")
        for layer_idx in range(tensor1.shape[0]):
            print("--------------------------------")
            layer_diff = abs_diff[layer_idx].max().item()
            print(f"Layer {layer_idx}: {layer_diff:.2e}")

            # 找到当前层最大绝对误差的索引
            layer_abs_diff = abs_diff[layer_idx]
            max_diff_idx = torch.argmax(layer_abs_diff.view(-1)).item()
            max_diff_indices = np.unravel_index(
                max_diff_idx, layer_abs_diff.shape
            )
            # 构建全局索引 (layer_idx, *local_indices)
            global_max_diff_indices = (layer_idx,) + max_diff_indices

            print(
                f"Maximum absolute difference at index: {global_max_diff_indices}"
            )

            # 显示最大误差位置的值
            jax_val = t1[global_max_diff_indices]
            torch_val = t2[global_max_diff_indices]
            diff_val = diff[global_max_diff_indices]

            print(f"JAX value at max diff location: {jax_val.item():.6f}")
            print(
                f"PyTorch value at max diff location: {torch_val.item():.6f}"
            )
            print(f"Difference at max location: {diff_val.item():.6e}")

            # 计算当前层的相对误差最大值（只考虑有效部分）
            layer_abs_diff = abs_diff[layer_idx]
            layer_ref_diff_tensor = ref_diff[layer_idx]
            valid_layer_ref_diff = layer_ref_diff_tensor[
                layer_abs_diff >= 1e-5
            ]
            if len(valid_layer_ref_diff) > 0:
                layer_ref_diff_max = valid_layer_ref_diff.max().item()
                print(
                    f"Layer {layer_idx} Reference difference: {layer_ref_diff_max:.2e} %"
                )
            else:
                print(
                    f"Layer {layer_idx} Reference difference: N/A (all abs diff < 1e-5)"
                )

            # 找到当前层最大相对误差的索引（只在有效位置）
            layer_ref_diff = ref_diff[layer_idx]
            layer_abs_diff = abs_diff[layer_idx]

            # 只考虑绝对误差 >= 1e-5 的位置
            valid_mask = layer_abs_diff >= 1e-5
            if valid_mask.any():
                # 在有效位置中找到最大相对误差
                valid_ref_diff_flat = layer_ref_diff.view(-1)[
                    valid_mask.view(-1)
                ]
                max_valid_ref_idx = torch.argmax(valid_ref_diff_flat).item()

                # 将平坦索引转换回原始形状的索引
                valid_indices = torch.nonzero(
                    valid_mask.view(-1), as_tuple=True
                )[0]
                max_ref_diff_idx = valid_indices[max_valid_ref_idx].item()

                max_ref_diff_indices = np.unravel_index(
                    max_ref_diff_idx, layer_ref_diff.shape
                )
                # 构建全局索引
                global_max_ref_diff_indices = (
                    layer_idx,
                ) + max_ref_diff_indices

                print(
                    f"Maximum relative difference at index: {global_max_ref_diff_indices}"
                )

                # 显示最大相对误差位置的值
                jax_val_ref = t1[global_max_ref_diff_indices]
                torch_val_ref = t2[global_max_ref_diff_indices]
                diff_val_ref = diff[global_max_ref_diff_indices]
                ref_diff_val = ref_diff[global_max_ref_diff_indices]

                print(
                    f"JAX value at max relative diff location: {jax_val_ref.item():.6f}"
                )
                print(
                    f"PyTorch value at max relative diff location: {torch_val_ref.item():.6f}"
                )
                print(f"Absolute difference: {diff_val_ref.item():.6e}")
                print(f"Relative difference: {ref_diff_val.item():.2e} %")
            else:
                print(
                    f"Maximum relative difference at index: N/A (all abs diff < 1e-5)"
                )

        # 找到误差最大的层
        layer_max_diffs = [
            abs_diff[layer_idx].max().item()
            for layer_idx in range(abs_diff.shape[0])
        ]
        worst_layer = torch.argmax(torch.tensor(layer_max_diffs)).item()
        print(
            f"\nWorst layer: {worst_layer} with max diff: {layer_max_diffs[worst_layer]:.2e}"
        )
    else:
        print("Tensor has only one dimension, skipping per-layer analysis")


def load_weights(filepath: str) -> Dict[str, torch.Tensor]:
    """Load weights from pickle file."""
    print(f"Loading weights from {filepath}")
    with open(filepath, "rb") as f:
        weights = pickle.load(f)
    return weights


def compare_tensor_shapes(
    tensor1: torch.Tensor, tensor2: torch.Tensor, key: str
) -> bool:
    """Compare shapes of two tensors."""
    if tensor1.shape != tensor2.shape:
        print(
            f"❌ Shape mismatch for {key}: {tensor1.shape} vs {tensor2.shape}"
        )
        return False
    return True


def compare_tensor_values(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    key: str,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare values of two tensors.

    Returns:
        (is_close, stats_dict) - stats_dict contains absolute and relative difference statistics
    """
    # Convert to float32 for comparison if needed
    t1 = tensor1.float()  # JAX weights
    t2 = tensor2.float()  # PyTorch weights

    # Create mask for elements that should be considered (absolute value >= atol)
    # Only compute differences for elements where at least one tensor has |value| >= atol
    significant_mask = (torch.abs(t1) >= atol) | (torch.abs(t2) >= atol)
    total_elements = t1.numel()

    # Calculate absolute differences, but set to 0 for insignificant elements
    raw_abs_diff = torch.abs(t1 - t2)
    abs_diff = torch.where(
        significant_mask, raw_abs_diff, torch.zeros_like(raw_abs_diff)
    )

    # 只在绝对误差 >= 1e-5 时计算相对误差（参考 debug_report.py）
    rel_diff = torch.where(
        abs_diff >= 1e-5,
        abs_diff * 100 / (torch.abs(t1) + 1e-6),
        torch.zeros_like(abs_diff),
    )

    # Calculate statistics only for significant elements
    significant_elements = significant_mask.sum().item()
    if significant_elements > 0:
        # Only consider differences of significant elements
        significant_abs_diff = abs_diff[significant_mask]
        significant_rel_diff = rel_diff[significant_mask]

        max_abs_diff = significant_abs_diff.max().item()
        mean_abs_diff = significant_abs_diff.mean().item()
        max_rel_diff = significant_rel_diff.abs().max().item()
        mean_rel_diff = significant_rel_diff.abs().mean().item()
    else:
        max_abs_diff = 0.0
        mean_abs_diff = 0.0
        max_rel_diff = 0.0
        mean_rel_diff = 0.0

    # Calculate percentiles for significant elements only
    if significant_elements > 0:
        significant_abs_diff_flat = abs_diff[significant_mask].flatten()
        significant_rel_diff_flat = rel_diff[significant_mask].flatten()

        abs_diff_sorted = torch.sort(significant_abs_diff_flat).values
        rel_diff_sorted = torch.sort(significant_rel_diff_flat.abs()).values

        # Percentiles based on significant elements
        percentiles = [50, 75, 90, 95, 99, 99.9]

        abs_percentiles = {}
        rel_percentiles = {}

        for p in percentiles:
            idx = int(significant_elements * p / 100)
            idx = min(idx, significant_elements - 1)
            abs_percentiles[f"p{p}"] = abs_diff_sorted[idx].item()
            rel_percentiles[f"p{p}"] = rel_diff_sorted[idx].item()
    else:
        abs_percentiles = {f"p{p}": 0.0 for p in [50, 75, 90, 95, 99, 99.9]}
        rel_percentiles = {f"p{p}": 0.0 for p in [50, 75, 90, 95, 99, 99.9]}

    # Check if tensors are close
    is_close = torch.allclose(t1, t2, rtol=rtol, atol=atol)

    stats = {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "abs_percentiles": abs_percentiles,
        "rel_percentiles": rel_percentiles,
        "total_elements": total_elements,
        "significant_elements": significant_elements,
        "insignificant_elements": total_elements - significant_elements,
    }

    return is_close, stats


def compare_weights(
    jax_weights: Dict[str, torch.Tensor],
    torch_weights: Dict[str, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Compare all weights between JAX and PyTorch versions.

    Returns:
        (all_aligned, comparison_results)
    """
    all_aligned = True
    results = {}

    # Get all keys from both dictionaries
    jax_keys = set(jax_weights.keys())
    torch_keys = set(torch_weights.keys())

    # Check for missing keys
    missing_in_torch = jax_keys - torch_keys
    missing_in_jax = torch_keys - jax_keys

    if missing_in_torch:
        print(f"⚠️  Keys missing in PyTorch weights: {missing_in_torch}")
        all_aligned = False

    if missing_in_jax:
        print(f"⚠️  Keys missing in JAX weights: {missing_in_jax}")
        all_aligned = False

    # Compare common keys
    common_keys = jax_keys & torch_keys
    print(f"\nComparing {len(common_keys)} common weight keys...")

    for key in sorted(common_keys):
        jax_tensor = jax_weights[key]
        torch_tensor = torch_weights[key]

        # Compare shapes
        shape_match = compare_tensor_shapes(jax_tensor, torch_tensor, key)

        if not shape_match:
            all_aligned = False
            results[key] = {
                "shape_match": False,
                "value_match": False,
                "max_diff": None,
                "mean_diff": None,
            }
            continue

        # Compare values
        value_match, stats = compare_tensor_values(
            jax_tensor, torch_tensor, key, rtol, atol
        )

        results[key] = {
            "shape_match": True,
            "value_match": value_match,
            "stats": stats,
            "shape": jax_tensor.shape,
        }

        if not value_match:
            all_aligned = False
            print(
                f"  🔴 {key}: Value mismatch (max_abs={stats['max_abs_diff']:.2e}, max_rel={stats['max_rel_diff']:.2f}%)"
            )
            # Print detailed per-layer analysis for mismatched weights
            print_detailed_layer_analysis(jax_tensor, torch_tensor, key)
        elif stats["max_abs_diff"] > 1e-10:  # Show small differences too
            print(
                f"  🟡 {key}: Small difference (max_abs={stats['max_abs_diff']:.2e}, max_rel={stats['max_rel_diff']:.2f}%)"
            )

    return all_aligned, results


def print_summary(results: Dict[str, Dict[str, Any]], all_aligned: bool):
    """Print detailed comparison summary."""
    print("\n" + "=" * 80)
    print("WEIGHT COMPARISON SUMMARY")
    print("=" * 80)

    total_keys = len(results)
    shape_mismatches = sum(1 for r in results.values() if not r["shape_match"])
    value_mismatches = sum(
        1
        for r in results.values()
        if r["shape_match"] and not r["value_match"]
    )
    perfect_matches = sum(
        1 for r in results.values() if r["shape_match"] and r["value_match"]
    )

    print(f"Total weight keys compared: {total_keys}")
    print(f"Perfect matches (shape + values): {perfect_matches}")
    print(f"Shape mismatches: {shape_mismatches}")
    print(f"Value mismatches (same shape): {value_mismatches}")

    if all_aligned:
        print("✅ ALL WEIGHTS ARE ALIGNED!")
    else:
        print("❌ WEIGHTS ARE NOT FULLY ALIGNED!")

    # Show problematic keys
    if shape_mismatches > 0:
        print("\nShape mismatches:")
        for key, result in results.items():
            if not result["shape_match"]:
                print(f"  🔴 {key}: Shape mismatch")

    # Show max differences for value mismatches
    if value_mismatches > 0:
        print(f"\nValue mismatches ({value_mismatches} total):")
        value_problems = [
            (k, r)
            for k, r in results.items()
            if r["shape_match"] and not r["value_match"]
        ]
        # Sort by max_abs_diff descending
        value_problems.sort(
            key=lambda x: x[1]["stats"]["max_abs_diff"], reverse=True
        )
        for key, result in value_problems[
            :10
        ]:  # Show top 10 largest differences
            stats = result["stats"]
            print(
                f"  🔴 {key}: max_abs={stats['max_abs_diff']:.2e}, max_rel={stats['max_rel_diff']:.2f}%"
            )
        if len(value_problems) > 10:
            print(f"  ... and {len(value_problems) - 10} more")

        # Show detailed percentile statistics for all mismatched weights
        print(f"\nDetailed statistics for value mismatches:")
        all_abs_diffs = []
        all_rel_diffs = []

        for _, result in value_problems:
            stats = result["stats"]
            # Collect percentile data from all mismatched tensors
            for p in [50, 75, 90, 95, 99, 99.9]:
                all_abs_diffs.append(stats["abs_percentiles"][f"p{p}"])
                all_rel_diffs.append(stats["rel_percentiles"][f"p{p}"])

        if all_abs_diffs:
            all_abs_diffs.sort()
            all_rel_diffs.sort()
            total_samples = len(all_abs_diffs)

            print("Absolute difference percentiles across all mismatches:")
            percentiles = [50, 75, 90, 95, 99, 99.9]
            for p in percentiles:
                idx = int(total_samples * p / 100)
                idx = min(idx, total_samples - 1)
                print(f"  P{p:5.1f}%: {all_abs_diffs[idx]:.2e}")

            print("\nRelative difference percentiles across all mismatches:")
            for p in percentiles:
                idx = int(total_samples * p / 100)
                idx = min(idx, total_samples - 1)
                print(f"  P{p:5.1f}%: {all_rel_diffs[idx]:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Compare JAX and PyTorch converted weights for alignment"
    )
    parser.add_argument(
        "--jax_weights",
        type=str,
        required=True,
        help="Path to pickle file containing JAX-converted weights",
    )
    parser.add_argument(
        "--torch_weights",
        type=str,
        required=True,
        help="Path to pickle file containing PyTorch-converted weights",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for value comparison (default: 1e-5)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for value comparison (default: 1e-6)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save detailed markdown report (optional)",
    )

    args = parser.parse_args()

    try:
        # Load weights
        jax_weights = load_weights(args.jax_weights)
        torch_weights = load_weights(args.torch_weights)

        print(f"JAX weights: {len(jax_weights)} keys")
        print(f"PyTorch weights: {len(torch_weights)} keys")

        # Compare weights
        all_aligned, results = compare_weights(
            jax_weights, torch_weights, args.rtol, args.atol
        )

        # Print summary
        print_summary(results, all_aligned)

        # Generate and save detailed markdown report if requested
        if args.output:
            print(f"\nGenerating detailed markdown report to {args.output}...")
            report = generate_comparison_report(
                results,
                all_aligned,
                jax_weights,
                torch_weights,
                args.jax_weights,
                args.torch_weights,
            )

            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"✅ Report saved to {args.output}")

        # Exit with appropriate code
        sys.exit(0 if all_aligned else 1)

    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
