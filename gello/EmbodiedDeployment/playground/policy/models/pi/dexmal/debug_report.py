#!/usr/bin/env python3
"""
Debug report script for analyzing JAX and PyTorch weights.
Loads both formats and provides analysis tools.
"""

import argparse
import os
import pickle
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch


def load_jax_weights(jax_path: str):
    """Load JAX weights - copied from convert_from_jax.py"""
    try:
        import jax
        import orbax.checkpoint as ocp
    except ImportError as e:
        print(f"❌ Failed to import JAX/Orbax: {e}")
        print("Please install JAX and Orbax to load JAX weights")
        return None

    params_path = os.path.join(jax_path, "params")
    print(f"Loading JAX weights from {params_path}")

    if not os.path.exists(params_path):
        print(f"❌ JAX params path does not exist: {params_path}")
        return None

    try:
        single_device_sharding = jax.sharding.SingleDeviceSharding(
            jax.devices("cpu")[0]
        )

        with ocp.PyTreeCheckpointer() as ckptr:
            metadata = ckptr.metadata(params_path)
            params = ckptr.restore(
                params_path,
                ocp.args.PyTreeRestore(
                    item=metadata,
                    restore_args=jax.tree.map(
                        lambda _: ocp.ArrayRestoreArgs(
                            restore_type=jax.Array,
                            sharding=single_device_sharding,
                        ),
                        metadata,
                    ),
                    transforms={},
                ),
            )["params"]
            print(f"✅ Loaded JAX params with keys: {list(params.keys())}")
            return params
    except Exception as e:
        print(f"❌ Failed to load JAX weights: {e}")
        return None


def load_pytorch_weights(pytorch_path: str):
    """Load PyTorch weights - copied from convert_from_torch.py"""
    try:
        from safetensors import safe_open
    except ImportError as e:
        print(f"❌ Failed to import safetensors: {e}")
        print("Please install safetensors to load PyTorch weights")
        return None, None

    print(f"Loading PyTorch weights from {pytorch_path}")

    if not os.path.exists(pytorch_path):
        print(f"❌ PyTorch weights path does not exist: {pytorch_path}")
        return None, None

    try:
        state_dict = {}
        with safe_open(pytorch_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
            meta_data = f.metadata()
            print(f"✅ Loaded PyTorch weights with {len(state_dict)} keys")
            return state_dict, meta_data
    except Exception as e:
        print(f"❌ Failed to load PyTorch weights: {e}")
        return None, None


def load_pickle_weights(pickle_path: str):
    """Load weights from pickle file."""
    print(f"Loading pickle weights from {pickle_path}")
    if not os.path.exists(pickle_path):
        print(f"❌ Pickle file does not exist: {pickle_path}")
        return None

    try:
        with open(pickle_path, "rb") as f:
            weights = pickle.load(f)
        print(f"✅ Loaded pickle weights with {len(weights)} keys")
        return weights
    except Exception as e:
        print(f"❌ Failed to load pickle weights: {e}")
        return None


def check_diff(jax_weights_per_layer, torch_weights_per_layer, layer_name):

    # 计算差异
    diff = jax_weights_per_layer - torch_weights_per_layer
    abs_diff = torch.abs(diff)

    # 只在绝对误差 >= 1e-5 时计算相对误差
    ref_diff = torch.where(
        abs_diff >= 1e-5,
        abs_diff * 100 / (torch.abs(jax_weights_per_layer) + 1e-6),
        torch.zeros_like(abs_diff),
    )
    # 计算有效的相对误差最大值（排除为0的部分）
    valid_ref_diff = ref_diff[abs_diff >= 1e-5]
    if len(valid_ref_diff) > 0:
        print(
            f"{layer_name} Reference difference: {valid_ref_diff.max().item():.2e} %"
        )
    else:
        print("Reference difference: N/A (all abs diff < 1e-5)")

    # 找到最大误差
    max_diff = abs_diff.max().item()
    print(f"Maximum absolute difference: {max_diff:.2e}")

    # 检查每层的最大误差
    print("\nPer-layer maximum absolute differences:")
    for layer_idx in range(jax_weights_per_layer.shape[0]):
        print("--------------------------------")
        layer_diff = abs_diff[layer_idx].max().item()
        print(f"Layer {layer_idx}: {layer_diff:.2e}")

        # 找到当前层最大绝对误差的索引
        layer_abs_diff = abs_diff[layer_idx]
        max_diff_idx = torch.argmax(layer_abs_diff.view(-1)).item()
        max_diff_indices = np.unravel_index(max_diff_idx, layer_abs_diff.shape)
        # 构建全局索引 (layer_idx, *local_indices)
        global_max_diff_indices = (layer_idx,) + max_diff_indices

        print(
            f"Maximum absolute difference at index: {global_max_diff_indices}"
        )

        # 显示最大误差位置的值
        jax_val = jax_weights_per_layer[global_max_diff_indices]
        torch_val = torch_weights_per_layer[global_max_diff_indices]
        diff_val = diff[global_max_diff_indices]

        print(f"JAX value at max diff location: {jax_val.item():.6f}")
        print(f"PyTorch value at max diff location: {torch_val.item():.6f}")
        print(f"Difference at max location: {diff_val.item():.6e}")

        # 计算当前层的相对误差最大值（只考虑有效部分）
        layer_abs_diff = abs_diff[layer_idx]
        layer_ref_diff_tensor = ref_diff[layer_idx]
        valid_layer_ref_diff = layer_ref_diff_tensor[layer_abs_diff >= 1e-5]
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
            valid_ref_diff_flat = layer_ref_diff.view(-1)[valid_mask.view(-1)]
            max_valid_ref_idx = torch.argmax(valid_ref_diff_flat).item()

            # 将平坦索引转换回原始形状的索引
            valid_indices = torch.nonzero(valid_mask.view(-1), as_tuple=True)[
                0
            ]
            max_ref_diff_idx = valid_indices[max_valid_ref_idx].item()

            max_ref_diff_indices = np.unravel_index(
                max_ref_diff_idx, layer_ref_diff.shape
            )
            # 构建全局索引
            global_max_ref_diff_indices = (layer_idx,) + max_ref_diff_indices

            print(
                f"Maximum relative difference at index: {global_max_ref_diff_indices}"
            )

            # 显示最大相对误差位置的值
            jax_val_ref = jax_weights_per_layer[global_max_ref_diff_indices]
            torch_val_ref = torch_weights_per_layer[
                global_max_ref_diff_indices
            ]
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


def check_encoder_ffn_gate_w(
    jax_weights, torch_weights, jax_dexmal_weights, torch_dexmal_weights
):

    dexmal_jax_encoder_ffn_gate_w = jax_dexmal_weights["encoder_ffn_gate_w"]
    dexmal_torch_encoder_ffn_gate_w = torch_dexmal_weights[
        "encoder_ffn_gate_w"
    ]

    check_diff(
        dexmal_jax_encoder_ffn_gate_w,
        dexmal_torch_encoder_ffn_gate_w,
        "encoder_ffn_gate_w",
    )

    # jax -> encoder_ffn_gate_w
    rms_norm = jax_weights["PaliGemma"]["llm"]["layers"]["pre_ffw_norm"][
        "scale"
    ]["value"].astype("float32")
    w_gate = jax_weights["PaliGemma"]["llm"]["layers"]["mlp"]["gating_einsum"][
        "value"
    ][:, 0].astype("float32")

    w_gate *= 1 + rms_norm[:, :, None]
    w_up *= 1 + rms_norm[:, :, None]

    jax_weights_encoder_ffn_gate_w = torch.tensor(
        w_gate, dtype=torch.bfloat16, device="cpu"
    )

    # torch -> encoder_ffn_gate_w
    torch_weights_encoder_ffn_gate_w = torch.zeroslike(
        jax_weights_encoder_ffn_gate_w, dtype=torch.bfloat16, device="cpu"
    )
    for layer_idx in range(18):
        # MLP weights
        rms_norm = torch_weights[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{layer_idx}.post_attention_layernorm.weight"
        ]

        gate_weight = torch_weights[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{layer_idx}.mlp.gate_proj.weight"
        ].t()
        up_weight = torch_weights[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{layer_idx}.mlp.up_proj.weight"
        ].t()

        gate_weight = gate_weight * (1 + rms_norm[:, None])
        up_weight = up_weight * (1 + rms_norm[:, None])

        torch_weights_encoder_ffn_gate_w[layer_idx] = gate_weight

    return


def main():
    from xdeploy.common.debug_utils import debugpy_listen

    debugpy_listen()

    jax_path = "/mnt/phwfile/efm_t/zhuyangkun_tmp_need_del/model/publish/pi0_base_libero"
    torch_path = "/mnt/phwfile/efm_t/zhuyangkun_tmp_need_del/model/publish/pi0_base_libero/torch/model.safetensors"
    jax_weights = load_jax_weights(jax_path)
    torch_weights = load_pytorch_weights(torch_path)

    jax_dexmal_weights = load_pickle_weights(
        os.path.join(jax_path, "dexmal", "checkpoint_jax.pkl")
    )
    torch_dexmal_weights = load_pickle_weights(
        os.path.join(jax_path, "dexmal", "checkpoint_torch.pkl")
    )

    check_encoder_ffn_gate_w(
        jax_weights, torch_weights, jax_dexmal_weights, torch_dexmal_weights
    )


if __name__ == "__main__":
    main()
