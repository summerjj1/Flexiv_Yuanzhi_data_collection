import logging
import math
import time
from copy import deepcopy

import torch._dynamo

torch._dynamo.config.suppress_errors = True

# Add openpi to the path or pip install -e openpi/, so that we can import the models_pytorch package
import openpi.models.gemma as _gemma
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
import torch
import torch.nn.functional as F  # noqa: N812
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from safetensors.torch import load_model as safetensors_load_model
from torch import Tensor, nn


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError(
            "The time tensor is expected to be of shape `(batch_size, )`."
        )

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(
        0.0, 1.0, dimension // 2, dtype=dtype, device=device
    )
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(
                action_expert_config.width, action_expert_config.width
            )
            self.time_mlp_out = nn.Linear(
                action_expert_config.width, action_expert_config.width
            )
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(
                2 * action_expert_config.width, action_expert_config.width
            )
            self.action_time_mlp_out = nn.Linear(
                action_expert_config.width, action_expert_config.width
            )

        # NOTE: disable matmul precision and compile for baseline performance(zhuyangkun)
        torch.set_float32_matmul_precision("high")
        # print('Compiling sample_actions...')
        # self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
        # print('Compilation completed')

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if (
                not check.check_whether_transformers_replace_is_installed_correctly()
            ):
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = (
            True
        )
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = (
            True
        )
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = (
            True
        )

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = (
            False
        )
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = (
            False
        )
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = (
            False
        )

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func,
                *args,
                use_reentrant=False,
                preserve_rng_state=False,
                **kwargs,
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(
            observation, train=train
        )
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(
                lang_tokens
            )
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(
            att_masks, dtype=torch.bool, device=pad_masks.device
        )

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=timestep.device
        )
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(
            att_masks, dtype=embs.dtype, device=embs.device
        )
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        (
            suffix_embs,
            suffix_pad_masks,
            suffix_att_masks,
            adarms_cond,
        ) = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        ):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(
        self, device, observation, noise=None, num_steps=10
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (
                bsize,
                self.config.action_horizon,
                self.config.action_dim,
            )
            noise = self.sample_noise(actions_shape, device)

        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(
            prefix_pad_masks, prefix_att_masks
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(
            prefix_att_2d_masks
        )
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
            "eager"  # noqa: SLF001
        )

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        (
            suffix_embs,
            suffix_pad_masks,
            suffix_att_masks,
            adarms_cond,
        ) = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(
            suffix_pad_masks, suffix_att_masks
        )

        full_att_2d_masks = torch.cat(
            [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
        )

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = (
            prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        )

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(
            full_att_2d_masks
        )
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "eager"  # noqa: SLF001
        )

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


def debugpy_listen():
    import debugpy

    debugpy.listen(("0.0.0.0", 10092))
    print("Waiting for client to attach 10092...")
    debugpy.wait_for_client()


def setup_device():
    """设置计算设备"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def create_model():
    """创建PI0模型"""
    from openpi.models.pi0_config import Pi0Config

    model = PI0Pytorch(Pi0Config)
    return model


def load_model_weights(model, model_path):
    """加载模型权重"""
    print("Loading model weights...")
    missing, unexpected = safetensors_load_model(model, model_path)
    print("Model weights loaded successfully")
    return missing, unexpected


def create_fake_observation(
    batch_size=1, image_size=224, prompt_seq_len=10, device=None
):
    """创建假的观察数据用于测试"""
    from collections import namedtuple

    from openpi.models.pi0_config import Pi0Config

    if device is None:
        device = setup_device()

    Observation = namedtuple(
        "Observation",
        [
            "images",
            "image_masks",
            "tokenized_prompt",
            "tokenized_prompt_mask",
            "state",
            "token_ar_mask",
            "token_loss_mask",
        ],
    )

    # 图像键
    image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

    # 创建假图像数据
    dummy_images = {
        key: torch.randn(
            batch_size,
            3,
            image_size,
            image_size,
            device=device,
            dtype=torch.bfloat16,
        )
        for key in image_keys
    }

    # 创建图像掩码
    dummy_image_masks = {
        key: torch.ones(batch_size, device=device, dtype=torch.bool)
        for key in image_keys
    }

    # 创建假的语言提示和掩码
    dummy_prompt = torch.randint(
        0, 1000, (batch_size, prompt_seq_len), device=device
    )  # tokenized prompt index
    dummy_prompt_mask = torch.ones(
        batch_size, prompt_seq_len, device=device, dtype=torch.bool
    )
    dummy_token_ar_mask = None  # Only for Pi Fast
    dummy_token_loss_mask = None  # Only for Pi Fast

    # 创建假状态张量
    dummy_state = torch.randn(
        batch_size, 32, device=device, dtype=torch.bfloat16
    )

    observation = Observation(
        images=dummy_images,
        image_masks=dummy_image_masks,
        tokenized_prompt=dummy_prompt,
        tokenized_prompt_mask=dummy_prompt_mask,
        state=dummy_state,
        token_ar_mask=dummy_token_ar_mask,
        token_loss_mask=dummy_token_loss_mask,
    )

    return observation, image_keys


def run_inference(model, observation, device, num_steps=10):
    """运行推理并返回预测动作"""
    print("Running inference with dummy data...")
    print(f"Batch size: {observation.state.shape[0]}")
    from openpi.models.pi0_config import Pi0Config

    print(f"Action horizon: {Pi0Config.action_horizon}")
    print(f"Action dimension: {Pi0Config.action_dim}")
    print(f"Image keys: {list(observation.images.keys())}")

    try:
        predicted_actions = model.sample_actions(
            device=device, observation=observation, num_steps=num_steps
        )
        print("\nInference successful!")
        return predicted_actions
    except Exception as e:
        print(f"\nAn error occurred during inference: {e}")
        print(
            "This might be due to missing dependencies (like 'gemma.py') or configuration mismatches."
        )
        raise


def verify_output_shape(predicted_actions, batch_size):
    """验证输出形状"""
    from openpi.models.pi0_config import Pi0Config

    print(f"Predicted actions shape: {predicted_actions.shape}")

    expected_shape = (
        batch_size,
        Pi0Config.action_horizon,
        Pi0Config.action_dim,
    )
    print(f"Expected shape: {expected_shape}")

    assert (
        predicted_actions.shape == expected_shape
    ), f"Shape mismatch! Expected {expected_shape}, got {predicted_actions.shape}"
    print("Output shape is correct.")


def model_init(model_path=None, device=None):
    """向后兼容的函数，初始化模型"""
    model = create_model()
    if model_path is not None:
        load_model_weights(model, model_path)
    model.to(device).eval()
    return model


def benchmark_model(
    model, observation, device, num_iterations=100, num_warmup=3
):
    """对模型进行性能基准测试"""
    print("\n=== Starting Benchmark ===")

    # Warm up - 预热阶段
    print(f"Warming up with {num_warmup} iterations...")
    for i in range(num_warmup):
        _ = model.sample_actions(
            device=device, observation=observation, num_steps=10
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        if (i + 1) % 1 == 0:
            print(f"Warmup iteration {i + 1}/{num_warmup} completed")

    print("Warmup completed. Starting benchmark...")

    # Benchmark - 基准测试阶段
    times = []
    for i in range(num_iterations):
        t0 = time.time()
        _ = model.sample_actions(
            device=device, observation=observation, num_steps=10
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.time()
        times.append(t1 - t0)

        if (i + 1) % 10 == 0:
            print(f"Benchmark iteration {i + 1}/{num_iterations} completed")

    # 计算统计信息
    times_ms = [t * 1000 for t in times]
    median_time = sorted(times_ms)[len(times_ms) // 2]
    mean_time = sum(times_ms) / len(times_ms)
    min_time = min(times_ms)
    max_time = max(times_ms)

    print("\n=== Benchmark Results ===")
    print(f"Device: {device}")
    print(f"Batch size: {observation.state.shape[0]}")
    print(f"Total iterations: {num_iterations}")
    print(f"Median time per inference: {median_time:.3f} ms")
    print(f"Mean time per inference: {mean_time:.3f} ms")
    print(f"Min time per inference: {min_time:.3f} ms")
    print(f"Max time per inference: {max_time:.3f} ms")

    # 计算FPS
    fps = 1000.0 / median_time
    print(f"Inference speed: {fps:.2f} FPS")

    return {
        "median_time_ms": median_time,
        "mean_time_ms": mean_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "fps": fps,
        "times_ms": times_ms,
    }


def main():
    """主函数：执行完整的模型推理流程"""
    # 调试模式（可选）
    # debugpy_listen()

    # 模型路径
    model_path = "/mnt/petrelfs/zhuyangkun/workspace/openpi/test.safetensors"
    model_path = (
        None  # "/mnt/petrelfs/zhuyangkun/workspace/openpi/test.safetensors"
    )

    # Step 1: 设置设备
    device = setup_device()
    model = model_init(model_path, device)

    # # 可选：编译模型以提高性能

    print("Compiling model...")
    compile_args = {
        "mode": "max-autotune",
        # 'dynamic': False,  # 静态形状通常更快
        # 'fullgraph': False,  # 启用完整图优化
        # 'backend': "tvm",  # 使用inductor后端
        # 'options': {
        #     "max_autotune": True,
        #     "triton.cudagraphs": True,  # 启用CUDA图
        #     "trace.enabled": False,
        #     "trace.graph_diagram": False,
        # }
    }
    print(f"Compiling model with args: {compile_args}")
    model.sample_actions = torch.compile(model.sample_actions, **compile_args)
    print("Model compilation completed")

    # Step 2: 创建假的输入数据
    batch_size = 1
    observation, image_keys = create_fake_observation(
        batch_size=batch_size, device=device
    )

    # Step 3: 运行推理
    predicted_actions = run_inference(model, observation, device, num_steps=10)

    # Step 4: 验证输出形状
    verify_output_shape(predicted_actions, batch_size)

    # Step 5: 性能基准测试
    print("=" * 100)
    print("Benchmarking model...")
    benchmark_results = benchmark_model(
        model, observation, device, num_iterations=100, num_warmup=3
    )
    print("=" * 100)

    print("\nAll steps completed successfully!")


if __name__ == "__main__":
    main()
