import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer


def convert_weights_from_pytorch(weights, pytorch_state_dict):
    """Convert PyTorch weights to custom format used by dexmal models."""

    # vision encoder weights
    # From paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding
    weights["vision_patch_embedding_w"].copy_(
        torch.tensor(
            pytorch_state_dict[
                "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
            ],
            dtype=torch.bfloat16,
            device="cpu",
        ).permute(
            2, 3, 1, 0
        )  # Convert from (out_channels, in_channels, kH, kW) to (kH, kW, in_channels, out_channels)
    )
    weights["vision_patch_embedding_b"].copy_(
        torch.tensor(
            pytorch_state_dict[
                "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
            ],
            dtype=torch.bfloat16,
            device="cpu",
        )
    )

    # Position embeddings
    weights["vision_position_embedding"].copy_(
        torch.tensor(
            pytorch_state_dict[
                "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
            ],
            dtype=torch.bfloat16,
            device="cpu",
        )
    )

    # Vision attention weights - need to split QKV
    num_layers = 27  # PaliGemma vision layers
    for layer_idx in range(num_layers):
        # QKV weights are concatenated in transformers format: [q, k, v]
        qkv_weight = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.weight"
        ]  # 1152, 1152
        k_weight = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.k_proj.weight"
        ]  # 1152, 1152
        v_weight = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.weight"
        ]  # 1152, 1152

        # Concatenate Q, K, V weights
        qkv_combined = torch.cat(
            [qkv_weight, k_weight, v_weight], dim=0
        )  # 3456, 1152
        weights["vision_attn_qkv_w"][layer_idx].copy_(
            qkv_combined.t()
        )  # (27, 1152, 16, 72) * 3(qkv)

        # QKV biases
        q_bias = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.bias"
        ]
        k_bias = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.k_proj.bias"
        ]
        v_bias = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.bias"
        ]

        qkv_bias_combined = torch.cat([q_bias, k_bias, v_bias], dim=0)
        weights["vision_attn_qkv_b"][layer_idx].copy_(qkv_bias_combined)

        # Attention output weights
        attn_o_weight = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.out_proj.weight"
        ]
        weights["vision_attn_o_w"][layer_idx].copy_(attn_o_weight.t())
        attn_o_bias = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn.out_proj.bias"
        ]
        weights["vision_attn_o_b"][layer_idx].copy_(attn_o_bias)

        # MLP weights
        mlp_gate_weight = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer_idx}.mlp.fc1.weight"
        ]
        weights["vision_ffn_up_w"][layer_idx].copy_(mlp_gate_weight.t())
        mlp_gate_bias = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer_idx}.mlp.fc1.bias"
        ]
        weights["vision_ffn_up_b"][layer_idx].copy_(mlp_gate_bias)

        mlp_down_weight = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer_idx}.mlp.fc2.weight"
        ]
        weights["vision_ffn_down_w"][layer_idx].copy_(mlp_down_weight.t())
        mlp_down_bias = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer_idx}.mlp.fc2.bias"
        ]
        weights["vision_ffn_down_b"][layer_idx].copy_(mlp_down_bias)

        # Layer norms
        weights["vision_pre_attn_norm_w"][layer_idx].copy_(
            torch.tensor(
                pytorch_state_dict[
                    f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer_idx}.layer_norm1.weight"
                ],
                dtype=torch.bfloat16,
                device="cpu",
            )
        )
        weights["vision_pre_attn_norm_b"][layer_idx].copy_(
            torch.tensor(
                pytorch_state_dict[
                    f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer_idx}.layer_norm1.bias"
                ],
                dtype=torch.bfloat16,
                device="cpu",
            )
        )
        weights["vision_pre_ffn_norm_w"][layer_idx].copy_(
            torch.tensor(
                pytorch_state_dict[
                    f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer_idx}.layer_norm2.weight"
                ],
                dtype=torch.bfloat16,
                device="cpu",
            )
        )
        weights["vision_pre_ffn_norm_b"][layer_idx].copy_(
            torch.tensor(
                pytorch_state_dict[
                    f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer_idx}.layer_norm2.bias"
                ],
                dtype=torch.bfloat16,
                device="cpu",
            )
        )

    # Final vision norm
    weights["vision_final_norm_w"].copy_(
        torch.tensor(
            pytorch_state_dict[
                "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
            ],
            dtype=torch.bfloat16,
            device="cpu",
        )
    )
    weights["vision_final_norm_b"].copy_(
        torch.tensor(
            pytorch_state_dict[
                "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"
            ],
            dtype=torch.bfloat16,
            device="cpu",
        )
    )

    # Multi-modal projector
    weights["encoder_multi_modal_projector_w"].copy_(
        torch.tensor(
            pytorch_state_dict[
                "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
            ],
            dtype=torch.bfloat16,
            device="cpu",
        ).t()
    )
    weights["encoder_multi_modal_projector_b"].copy_(
        torch.tensor(
            pytorch_state_dict[
                "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"
            ],
            dtype=torch.bfloat16,
            device="cpu",
        )
    )

    # Language model (Gemma) - encoder
    num_text_layers = 18
    for layer_idx in range(num_text_layers):
        # Self-attention weights - QKV
        q_weight = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{layer_idx}.self_attn.q_proj.weight"
        ].t()  # 这里有提前转置，避免后续计算数据重排列耗时
        k_weight = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{layer_idx}.self_attn.k_proj.weight"
        ].t()
        v_weight = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{layer_idx}.self_attn.v_proj.weight"
        ].t()  # 256, 2048

        w_scale = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{layer_idx}.input_layernorm.weight"
        ]  # 2048

        q_weight = q_weight * (1 + w_scale[:, None])
        k_weight = k_weight * (1 + w_scale[:, None])
        v_weight = v_weight * (1 + w_scale[:, None])

        # Combine QKV and reshape for JAX format
        q_combined = (
            q_weight.reshape(2048, 8, 2, 128)
            .permute(0, 1, 3, 2)
            .reshape(2048, 2048)
        )
        k_combined = (
            k_weight.reshape(2048, 2, 128).permute(0, 2, 1).reshape(2048, 256)
        )
        v_combined = v_weight

        # Store in encoder format (similar to JAX)
        weights["encoder_attn_qkv_w"][layer_idx] = torch.cat(
            [q_combined, k_combined, v_combined], dim=1
        )

        # Attention output
        o_weight = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{layer_idx}.self_attn.o_proj.weight"
        ]
        weights["encoder_attn_o_w"][layer_idx] = o_weight.t().reshape(
            2048, 2048
        )

        # MLP weights
        rms_norm = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{layer_idx}.post_attention_layernorm.weight"
        ]

        gate_weight = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{layer_idx}.mlp.gate_proj.weight"
        ].t()
        up_weight = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{layer_idx}.mlp.up_proj.weight"
        ].t()
        down_weight = pytorch_state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{layer_idx}.mlp.down_proj.weight"
        ].t()

        gate_weight = gate_weight * (1 + rms_norm[:, None])
        up_weight = up_weight * (1 + rms_norm[:, None])

        weights["encoder_ffn_gate_w"][layer_idx] = gate_weight
        weights["encoder_ffn_up_w"][layer_idx] = up_weight
        weights["encoder_ffn_down_w"][layer_idx] = down_weight

    # Decoder (action expert) weights
    for layer_idx in range(num_text_layers):
        # Decoder attention QKV
        q_weight = pytorch_state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{layer_idx}.self_attn.q_proj.weight"
        ].t()
        k_weight = pytorch_state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{layer_idx}.self_attn.k_proj.weight"
        ].t()
        v_weight = pytorch_state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{layer_idx}.self_attn.v_proj.weight"
        ].t()
        w_scale = pytorch_state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{layer_idx}.input_layernorm.weight"
        ]

        q_weight = q_weight * (1 + w_scale[:, None])
        k_weight = k_weight * (1 + w_scale[:, None])
        v_weight = v_weight * (1 + w_scale[:, None])

        # Combine and reshape for decoder format
        q_combined = (
            q_weight.reshape(1024, 8, 2, 128)
            .permute(0, 1, 3, 2)
            .reshape(1024, 2048)
        )
        k_combined = (
            k_weight.reshape(1024, 2, 128).permute(0, 2, 1).reshape(1024, 256)
        )
        v_combined = v_weight

        weights["decoder_attn_qkv_w"][layer_idx] = torch.cat(
            [q_combined, k_combined, v_combined], dim=1
        )

        # Decoder attention output
        o_weight = pytorch_state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{layer_idx}.self_attn.o_proj.weight"
        ]
        weights["decoder_attn_o_w"][layer_idx] = o_weight.t().reshape(
            2048, 1024
        )

        # Decoder MLP
        rms_norm = pytorch_state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{layer_idx}.post_attention_layernorm.weight"
        ]
        gate_weight = pytorch_state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{layer_idx}.mlp.gate_proj.weight"
        ].t()
        up_weight = pytorch_state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{layer_idx}.mlp.up_proj.weight"
        ].t()
        down_weight = pytorch_state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{layer_idx}.mlp.down_proj.weight"
        ].t()

        gate_weight = gate_weight * (1 + rms_norm[:, None])
        up_weight = up_weight * (1 + rms_norm[:, None])

        weights["decoder_ffn_gate_w"][layer_idx] = gate_weight
        weights["decoder_ffn_up_w"][layer_idx] = up_weight
        weights["decoder_ffn_down_w"][layer_idx] = down_weight

    # Decoder state input projection
    weights["decoder_state_in_proj_w"].copy_(
        torch.tensor(
            pytorch_state_dict["state_proj.weight"].t(),
            dtype=torch.bfloat16,
            device="cpu",
        )
    )
    weights["decoder_state_in_proj_b"].copy_(
        torch.tensor(
            pytorch_state_dict.get("state_proj.bias", torch.zeros(1024)),
            dtype=torch.bfloat16,
            device="cpu",
        )
    )

    def _create_sinusoidal_pos_embedding(
        time: torch.tensor,
        dimension: int,
        min_period: float,
        max_period: float,
        device="cpu",
    ):
        dtype = torch.float32
        fraction = torch.linspace(
            0.0, 1.0, dimension // 2, dtype=dtype, device=device
        )
        period = min_period * (max_period / min_period) ** fraction
        scaling_factor = 1.0 / period * 2 * torch.pi
        sin_input = scaling_factor[None, :] * time[:, None]
        pos_emb = torch.cat(
            [torch.sin(sin_input), torch.cos(sin_input)], dim=1
        )
        return pos_emb

    n_decode_steps = 10
    # Extract action-time MLP weights from PyTorch state dict
    action_time_mlp_in_weight = pytorch_state_dict[
        "action_time_mlp_in.weight"
    ].t()
    mlp_in_weight_action = action_time_mlp_in_weight[:1024, :]
    mlp_in_weight_time = action_time_mlp_in_weight[1024:, :]

    action_time_mlp_in_bias = pytorch_state_dict["action_time_mlp_in.bias"]
    action_in_proj_weight = pytorch_state_dict["action_in_proj.weight"].t()
    action_in_proj_bias = pytorch_state_dict["action_in_proj.bias"]

    decoder_action_fused_out_proj_w = pytorch_state_dict[
        "action_out_proj.weight"
    ].t()
    decoder_action_fused_out_proj_b = pytorch_state_dict[
        "action_out_proj.bias"
    ]

    # Get final norm scale from decoder
    final_norm_scale = pytorch_state_dict[
        "paligemma_with_expert.gemma_expert.model.norm.weight"
    ]

    fused_weight = torch.matmul(action_in_proj_weight, mlp_in_weight_action)
    action_bias_contrib = torch.matmul(
        mlp_in_weight_action.T, action_in_proj_bias
    )
    time_dependent_biases = torch.zeros(
        n_decode_steps, 1024, device="cpu", dtype=torch.bfloat16
    )
    for t in range(n_decode_steps):
        time_val = 1.0 - t / n_decode_steps
        time_tensor = torch.tensor([time_val], device="cpu")
        time_emb = _create_sinusoidal_pos_embedding(
            time_tensor, 1024, 4e-3, 4.0, "cpu"
        ).squeeze(0)
        time_emb = time_emb.to(torch.bfloat16)
        time_contrib = torch.matmul(mlp_in_weight_time.T, time_emb)
        time_dependent_biases[t] = (
            action_bias_contrib + time_contrib + action_time_mlp_in_bias
        ).to(torch.bfloat16)

    weights["decoder_action_mlp_w"].copy_(
        torch.tensor(
            pytorch_state_dict["action_time_mlp_out.weight"],
            dtype=torch.bfloat16,
            device="cpu",
        ).t()
    )
    weights["decoder_action_mlp_b"].copy_(
        torch.tensor(
            pytorch_state_dict["action_time_mlp_out.bias"],
            dtype=torch.bfloat16,
            device="cpu",
        )
    )

    decoder_action_fused_out_proj_w *= 1 + final_norm_scale[:, None]
    decoder_action_fused_out_proj_w *= -0.1
    decoder_action_fused_out_proj_b *= -0.1
    weights["decoder_action_fused_in_proj_w"].copy_(fused_weight)
    weights["decoder_action_fused_time_biases"].copy_(time_dependent_biases)
    weights["decoder_action_fused_out_proj_w"].copy_(
        decoder_action_fused_out_proj_w
    )
    weights["decoder_action_fused_out_proj_b"].copy_(
        decoder_action_fused_out_proj_b
    )


def load_pytorch_weights(pytorch_path: str):
    """Load PyTorch weights from safetensors format."""
    from safetensors import safe_open

    print(f"Loading PyTorch weights from {pytorch_path}")

    # Load safetensors file
    state_dict = {}
    with safe_open(pytorch_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)

        meta_data = f.metadata()

    return state_dict, meta_data


def prepare_prompt(prompt: str, embedding_weight, tokenizer_path):
    """Prepare prompt embeddings (same as in convert_from_jax.py)."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    embedding_weight_torch = torch.tensor(
        embedding_weight, dtype=torch.bfloat16, device="cpu"
    )
    num_embeddings, embedding_dim = embedding_weight_torch.shape
    language_embedding = (
        nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        .bfloat16()
        .cpu()
    )
    with torch.no_grad():
        language_embedding.weight.copy_(embedding_weight_torch)
    prompt = [prompt.strip().replace("_", " ") + "\n"]
    language_tokens = (
        tokenizer(
            prompt,
            max_length=48,
            return_tensors="pt",
        )["input_ids"]
        .cpu()
        .squeeze(0)
    )
    language_embeds = language_embedding(language_tokens)
    language_embeds *= language_embeds.shape[-1] ** 0.5
    return language_embeds, language_embeds.shape[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert PyTorch weights to custom format"
    )
    parser.add_argument(
        "--pytorch_path",
        type=str,
        required=True,
        help="Path to PyTorch safetensors file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for converted weights",
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Prompt text"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to tokenizer"
    )
    args = parser.parse_args()

    # from xdeploy.common.debug_utils import debugpy_listen
    # debugpy_listen()

    # Load PyTorch weights #
    # If the weights are from OpenPI's pi0, they need to be converted to the Dexmal format.
    # Because safetensors will remove some shared storage keys, the keys are stored in meta_data.
    pytorch_state_dict, meta_data = load_pytorch_weights(args.pytorch_path)

    # Extract language embedding for prompt preparation
    embedding_weight = pytorch_state_dict[
        meta_data[
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        ]
    ]
    language_embeds, prompt_len = prepare_prompt(
        args.prompt, embedding_weight, args.tokenizer_path
    )

    # Initialize weights dictionary (same structure as convert_from_jax.py)
    # Weight folding, parameter folding
    weights = {
        # Vision encoder weights
        "vision_patch_embedding_w": torch.zeros(
            14, 14, 3, 1152, dtype=torch.bfloat16, device="cpu"
        ),
        "vision_patch_embedding_b": torch.zeros(
            1152, dtype=torch.bfloat16, device="cpu"
        ),
        "vision_position_embedding": torch.zeros(
            256, 1152, dtype=torch.bfloat16, device="cpu"
        ),
        "vision_attn_qkv_w": torch.zeros(
            27, 1152, 3 * 1152, dtype=torch.bfloat16, device="cpu"
        ),
        "vision_attn_qkv_b": torch.zeros(
            27, 3 * 1152, dtype=torch.bfloat16, device="cpu"
        ),
        "vision_attn_o_w": torch.zeros(
            27, 1152, 1152, dtype=torch.bfloat16, device="cpu"
        ),
        "vision_attn_o_b": torch.zeros(
            27, 1152, dtype=torch.bfloat16, device="cpu"
        ),
        "vision_ffn_up_w": torch.zeros(
            27, 1152, 4304, dtype=torch.bfloat16, device="cpu"
        ),
        "vision_ffn_up_b": torch.zeros(
            27, 4304, dtype=torch.bfloat16, device="cpu"
        ),
        "vision_ffn_down_w": torch.zeros(
            27, 4304, 1152, dtype=torch.bfloat16, device="cpu"
        ),
        "vision_ffn_down_b": torch.zeros(
            27, 1152, dtype=torch.bfloat16, device="cpu"
        ),
        "vision_pre_attn_norm_w": torch.zeros(
            27, 1152, dtype=torch.bfloat16, device="cpu"
        ),
        "vision_pre_attn_norm_b": torch.zeros(
            27, 1152, dtype=torch.bfloat16, device="cpu"
        ),
        "vision_pre_ffn_norm_w": torch.zeros(
            27, 1152, dtype=torch.bfloat16, device="cpu"
        ),
        "vision_pre_ffn_norm_b": torch.zeros(
            27, 1152, dtype=torch.bfloat16, device="cpu"
        ),
        "vision_final_norm_w": torch.zeros(
            1152, dtype=torch.bfloat16, device="cpu"
        ),
        "vision_final_norm_b": torch.zeros(
            1152, dtype=torch.bfloat16, device="cpu"
        ),
        # Encoder weights
        "encoder_multi_modal_projector_w": torch.zeros(
            1152, 2048, dtype=torch.bfloat16, device="cpu"
        ),
        "encoder_multi_modal_projector_b": torch.zeros(
            2048, dtype=torch.bfloat16, device="cpu"
        ),
        "encoder_attn_qkv_w": torch.zeros(
            18, 2048, 2560, dtype=torch.bfloat16, device="cpu"
        ),
        "encoder_attn_o_w": torch.zeros(
            18, 2048, 2048, dtype=torch.bfloat16, device="cpu"
        ),
        "encoder_ffn_gate_w": torch.zeros(
            18, 2048, 16384, dtype=torch.bfloat16, device="cpu"
        ),
        "encoder_ffn_up_w": torch.zeros(
            18, 2048, 16384, dtype=torch.bfloat16, device="cpu"
        ),
        "encoder_ffn_down_w": torch.zeros(
            18, 16384, 2048, dtype=torch.bfloat16, device="cpu"
        ),
        # Decoder weights
        "decoder_state_in_proj_w": torch.zeros(
            32, 1024, dtype=torch.bfloat16, device="cpu"
        ),
        "decoder_state_in_proj_b": torch.zeros(
            1024, dtype=torch.bfloat16, device="cpu"
        ),
        "decoder_action_fused_in_proj_w": torch.zeros(
            32, 1024, dtype=torch.bfloat16, device="cpu"
        ),
        "decoder_action_fused_time_biases": torch.zeros(
            10, 1024, dtype=torch.bfloat16, device="cpu"
        ),
        "decoder_action_mlp_w": torch.zeros(
            1024, 1024, dtype=torch.bfloat16, device="cpu"
        ),
        "decoder_action_mlp_b": torch.zeros(
            1024, dtype=torch.bfloat16, device="cpu"
        ),
        "decoder_attn_qkv_w": torch.zeros(
            18, 1024, 2560, dtype=torch.bfloat16, device="cpu"
        ),
        "decoder_attn_o_w": torch.zeros(
            18, 2048, 1024, dtype=torch.bfloat16, device="cpu"
        ),
        "decoder_ffn_gate_w": torch.zeros(
            18, 1024, 4096, dtype=torch.bfloat16, device="cpu"
        ),
        "decoder_ffn_up_w": torch.zeros(
            18, 1024, 4096, dtype=torch.bfloat16, device="cpu"
        ),
        "decoder_ffn_down_w": torch.zeros(
            18, 4096, 1024, dtype=torch.bfloat16, device="cpu"
        ),
        "decoder_action_fused_out_proj_w": torch.zeros(
            1024, 32, dtype=torch.bfloat16, device="cpu"
        ),
        "decoder_action_fused_out_proj_b": torch.zeros(
            32, dtype=torch.bfloat16, device="cpu"
        ),
        # Language embeds
        "language_embeds": torch.zeros(
            prompt_len, 2048, dtype=torch.bfloat16, device="cpu"
        ),
    }

    # Convert weights from PyTorch format
    convert_weights_from_pytorch(weights, pytorch_state_dict)
    weights["language_embeds"].copy_(language_embeds)

    # Save converted weights
    with open(args.output, "wb") as f:
        pickle.dump(weights, f)
    print(f"Converted weights saved to {args.output}")
