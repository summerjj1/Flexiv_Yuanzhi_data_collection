from __future__ import annotations

import argparse
import base64
import os
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Mapping, Optional

import numpy as np


def _encode_np(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        data_b64 = base64.b64encode(obj.tobytes(order="C")).decode("ascii")
        return {
            "__ndarray__": True,
            "dtype": str(obj.dtype),
            "shape": obj.shape,
            "data_b64": data_b64,
        }
    return obj


def _decode_np(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get("__ndarray__") is True:
        dtype = np.dtype(obj["dtype"])
        shape = tuple(obj["shape"])
        raw = base64.b64decode(obj["data_b64"].encode("ascii"))
        return np.frombuffer(raw, dtype=dtype).reshape(shape)
    return obj


def _to_hwc_uint8(img: Any) -> Optional[np.ndarray]:
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.size == 0:
        return None
    # Expect HWC
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr
    return None


def _to_thwc_uint8(stack: Any) -> Optional[np.ndarray]:
    """Accept (T,H,W,C) uint8 stacks with T=2 from client-side history."""
    if stack is None:
        return None
    arr = np.asarray(stack)
    if arr.size == 0:
        return None
    if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr
    # Backward compatibility: allow single frame HWC and repeat
    hwc = _to_hwc_uint8(arr)
    if hwc is not None:
        return np.stack([hwc, hwc], axis=0)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QwenA1 ZMQ policy server (ROUTER/DEALER, msgpack + numpy)."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--ckpt-path",
        required=True,
        help="Path to lerobot pretrained_model directory containing stats.json.",
    )
    parser.add_argument(
        "--encoder-path",
        default=None,
        help=(
            "Override Cosmos tokenizer encoder.jit filepath used inside lerobot "
            "(e.g. /workspace/checkpoints/nvidia/Cosmos-Tokenizer-CI8x8/encoder.jit). "
            "If provided and a sibling decoder.jit exists, server will also use it."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "bfloat16"),
        help="Inference dtype.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, e.g. cuda/cpu.",
    )
    parser.add_argument(
        "--action-mode",
        default="delta",
        choices=("delta", "abs"),
        help="Whether to treat predicted joints as delta or absolute.",
    )
    # History is maintained on client side; server expects 2-frame stacks.
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="Warmup steps for first request per process.",
    )
    return parser.parse_args()


class _QwenA1Engine:
    def __init__(
        self,
        ckpt_path: Path,
        encoder_path: str | None,
        device: str,
        dtype: str,
        action_mode: str,
        warmup_steps: int,
    ) -> None:
        self._ckpt_path = ckpt_path
        self._encoder_path = encoder_path
        self._device = device
        self._dtype = dtype
        self._action_mode = action_mode
        self._warmup_steps = max(0, int(warmup_steps))
        self._did_warmup = False

        # Lazy imports: keep server importable even without lerobot installed.
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.datasets.utils import load_json
            from lerobot.policies.qwena1 import QwenA1Config, QwenA1Policy
            from lerobot.policies.qwena1.transform_qwena1 import (
                Qwen3_VLProcessorTransformFn,
            )
            from lerobot.transforms.core import (
                NormalizeTransformFn,
                ResizeImagesWithPadFn,
                UnNormalizeTransformFn,
                compose,
            )
            from lerobot.utils.constants import OBS_IMAGES
            from lerobot.policies.qwena1.cosmos_tokenizer.image_lib import (
                ImageTokenizer,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Missing lerobot/qwena1 dependencies. Ensure lerobot is installed and accessible. "
                f"Original error: {e}"
            ) from e

        self._torch = torch
        self._OBS_IMAGES = OBS_IMAGES

        # Optional override for Cosmos tokenizer JIT filepaths without modifying lerobot.
        if self._encoder_path:
            encoder_path = self._encoder_path

            # If caller only provides encoder.jit, try to auto-pair decoder.jit.
            decoder_path: str | None = None
            if encoder_path.endswith("encoder.jit"):
                candidate = encoder_path[: -len("encoder.jit")] + "decoder.jit"
                if os.path.exists(candidate):
                    decoder_path = candidate

            if not getattr(ImageTokenizer, "_xdeploy_patched", False):
                _orig_init = ImageTokenizer.__init__

                def _patched_init(  # type: ignore[no-redef]
                    self,
                    checkpoint: str = None,
                    checkpoint_enc: str = None,
                    checkpoint_dec: str = None,
                    tokenizer_config: dict[str, Any] = None,
                    device: str = "cuda",
                ) -> None:
                    # Always override encoder; override decoder only if we have a paired path.
                    checkpoint_enc = encoder_path
                    # lerobot's QwenA1 hardcodes a relative decoder path; if we have a sibling
                    # decoder.jit next to encoder.jit, always prefer it when encoder-path is set.
                    if decoder_path:
                        checkpoint_dec = decoder_path
                    _orig_init(
                        self,
                        checkpoint=checkpoint,
                        checkpoint_enc=checkpoint_enc,
                        checkpoint_dec=checkpoint_dec,
                        tokenizer_config=tokenizer_config,
                        device=device,
                    )

                ImageTokenizer.__init__ = _patched_init  # type: ignore[assignment]
                ImageTokenizer._xdeploy_patched = True  # type: ignore[attr-defined]

        config = PreTrainedConfig.from_pretrained(ckpt_path)
        if not isinstance(config, QwenA1Config):
            raise TypeError(
                f"Expected QwenA1Config, got {type(config).__name__}"
            )
        config.compile_model = True
        config.compile_mode = "reduce-overhead"

        policy = QwenA1Policy.from_pretrained(
            config=config,
            pretrained_name_or_path=ckpt_path,
        )
        policy.to(device=device)
        if dtype == "bfloat16":
            policy.to(dtype=torch.bfloat16)
        else:
            policy.to(dtype=torch.float32)
        policy.eval()

        # Normalization stats
        stats = load_json(ckpt_path / "stats.json")["a2d"]
        stat_keys = ["min", "max", "mean", "std"]
        state_stat = {
            stat_key: np.concatenate(
                [
                    stats["observation.states.joint.position"][stat_key],
                    stats["observation.states.effector.position"][stat_key],
                ],
                axis=-1,
            )
            for stat_key in stat_keys
        }
        state_stat = {"observation.state": state_stat}
        action_stat = {
            stat_key: np.concatenate(
                [
                    stats["actions.joint.position"][stat_key],
                    stats["actions.effector.position"][stat_key],
                ],
                axis=-1,
            )
            for stat_key in stat_keys
        }
        action_stat = {"action": action_stat}
        unnormalize_fn = UnNormalizeTransformFn(
            selected_keys=["action"],
            mode="mean_std",
            norm_stats=action_stat,
        )

        input_transforms = compose(
            [
                ResizeImagesWithPadFn(height=224, width=224),
                Qwen3_VLProcessorTransformFn(),
                NormalizeTransformFn(
                    selected_keys=["observation.state"], norm_stats=state_stat
                ),
            ]
        )

        self._policy = policy
        self._input_transforms = input_transforms
        self._unnormalize_fn = unnormalize_fn
        self._chunk_size = int(getattr(config, "chunk_size", 1) or 1)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def infer_action_chunk(
        self,
        *,
        task: str,
        state_16: np.ndarray,
        img0_thwc: np.ndarray,
        img1_thwc: np.ndarray,
        img2_thwc: np.ndarray,
    ) -> np.ndarray:
        torch = self._torch

        # Client already provides 2-frame history stacks (T,H,W,C). Convert to (T,C,H,W).
        def _stack_from_thwc(thwc: np.ndarray) -> "torch.Tensor":
            t = torch.from_numpy(np.asarray(thwc).copy()).to(device=self._device)
            t = t.float() / 255.0
            t_chw = t.permute(0, 3, 1, 2)  # (T,C,H,W)
            t_chw = torch.nn.functional.interpolate(
                t_chw,
                (480, 640),
                mode="bilinear",
                align_corners=False,
            )
            return t_chw

        image0 = _stack_from_thwc(img0_thwc)
        image1 = _stack_from_thwc(img1_thwc)
        image2 = _stack_from_thwc(img2_thwc)

        sample: Dict[str, Any] = {
            f"{self._OBS_IMAGES}.image0": image0,
            f"{self._OBS_IMAGES}.image1": image1,
            f"{self._OBS_IMAGES}.image2": image2,
            "observation.state": torch.from_numpy(state_16.astype(np.float32))
            .to(device=self._device)
            .float(),
            "task": task,
        }
        sample = self._input_transforms(sample)

        # Convert to inputs expected by QwenA1Policy
        dtype = torch.bfloat16 if self._dtype == "bfloat16" else torch.float32
        inputs: Dict[str, Any] = {}
        for key, value in sample.items():
            if key == "task":
                inputs[key] = [value]
            elif hasattr(value, "dtype") and value.dtype == torch.int64:
                inputs[key] = value[None].to(device=self._device)
            else:
                inputs[key] = value[None].to(device=self._device, dtype=dtype)
        inputs.update(
            {
                f"{self._OBS_IMAGES}.image0_mask": torch.tensor([True]).to(
                    device=self._device
                ),
                f"{self._OBS_IMAGES}.image1_mask": torch.tensor([True]).to(
                    device=self._device
                ),
                f"{self._OBS_IMAGES}.image2_mask": torch.tensor([True]).to(
                    device=self._device
                ),
            }
        )

        with torch.no_grad():
            if not self._did_warmup and self._warmup_steps > 0:
                for _ in range(self._warmup_steps):
                    _ = self._policy.predict_action_chunk(
                        inputs, decode_image=False
                    )
                self._did_warmup = True

            action_pred, _img_pred = self._policy.predict_action_chunk(
                inputs, decode_image=False
            )

        # action_pred: (B, T, D). Take first batch and first 16 dims.
        action_pred = action_pred[0, :, :16].clone()
        action_pred = self._unnormalize_fn({"action": action_pred})["action"]

        if self._action_mode == "delta":
            init_joint = torch.from_numpy(state_16[:14].astype(np.float32)).to(
                device=action_pred.device
            )
            action_pred[:, :14] += init_joint

        # Reorder for A2D controller: [L7, Lg, R7, Rg]
        # Model order assumed: [L7, R7, Lg, Rg]?? (infer_qwen_a1 uses 14 joints + 2 grippers)
        # Here we assume: first 14 are joints (L7+R7), last 2 are grippers (L,R).
        action_np = action_pred.to(torch.float32).cpu().numpy()
        joints14 = action_np[:, :14]
        grips2 = action_np[:, 14:16]
        left7 = joints14[:, :7]
        right7 = joints14[:, 7:14]
        leftg = grips2[:, 0:1]
        rightg = grips2[:, 1:2]
        a2d_action = np.concatenate([left7, leftg, right7, rightg], axis=1)
        return a2d_action.astype(np.float32, copy=False)


class _Session:
    def __init__(
        self,
        *,
        engine: _QwenA1Engine,
    ) -> None:
        self._engine = engine

    def infer_next_chunk(self, payload: Mapping[str, Any]) -> np.ndarray:
        images = payload.get("images")
        state = payload.get("state")
        task = payload.get("task") or payload.get("prompt") or ""

        if not isinstance(images, Mapping):
            raise ValueError("payload.images must be a mapping")
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_arr.shape[0] != 16:
            raise ValueError(f"payload.state must be shape (16,), got {state_arr.shape}")

        # Input keys are expected from config camera names.
        # Each image must be a 2-frame stack (T,H,W,C) from client-side history.
        img0 = _to_thwc_uint8(images.get("cam_high"))
        img1 = _to_thwc_uint8(images.get("cam_left_wrist"))
        img2 = _to_thwc_uint8(images.get("cam_right_wrist"))
        if img0 is None or img1 is None or img2 is None:
            raise ValueError(
                "Missing required images: cam_high/cam_left_wrist/cam_right_wrist"
            )

        actions = self._engine.infer_action_chunk(
            task=str(task),
            state_16=state_arr,
            img0_thwc=img0,
            img1_thwc=img1,
            img2_thwc=img2,
        )
        return actions


def _serve(args: argparse.Namespace) -> None:
    try:
        import msgpack
        import zmq
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Server requires pyzmq and msgpack: pip install pyzmq msgpack"
        ) from e

    engine = _QwenA1Engine(
        ckpt_path=Path(args.ckpt_path),
        encoder_path=args.encoder_path,
        device=args.device,
        dtype=args.dtype,
        action_mode=args.action_mode,
        warmup_steps=args.warmup_steps,
    )

    metadata = {
        "model": "qwena1",
        "chunk_size": engine.chunk_size,
        "ts": time.time(),
    }
    metadata_bytes = msgpack.packb(
        metadata, use_bin_type=True, default=_encode_np
    )

    endpoint = f"tcp://{args.host}:{args.port}"
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.ROUTER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(endpoint)
    print(f"[qwena1_server] listening on {endpoint}")

    # One session per client identity (to isolate any per-client state if needed).
    sessions: Dict[bytes, _Session] = {}

    def _pack_envelope(obj: Dict[str, Any]) -> bytes:
        return msgpack.packb(obj, use_bin_type=True, default=_encode_np)

    def _unpack_envelope(data: bytes) -> Dict[str, Any]:
        out = msgpack.unpackb(data, raw=False)
        if not isinstance(out, dict):
            raise ValueError(f"invalid envelope type: {type(out).__name__}")
        return out

    while True:
        frames = sock.recv_multipart()
        if not frames:
            continue

        ident = frames[0]
        data = frames[-1]

        # Get/create per-client session.
        session = sessions.get(ident)
        if session is None:
            session = _Session(engine=engine)
            sessions[ident] = session

        req_id = uuid.uuid4().hex
        try:
            env = _unpack_envelope(data)
            req_id = str(env.get("id") or req_id)
            msg_type = env.get("type")

            if msg_type == "hello":
                resp = {
                    "ok": True,
                    "type": "hello_ack",
                    "id": req_id,
                    "metadata": msgpack.unpackb(
                        metadata_bytes, raw=False, object_hook=_decode_np
                    ),
                }
                sock.send_multipart([ident, _pack_envelope(resp)])
                continue

            if msg_type != "infer":
                raise ValueError(f"unknown request type: {msg_type}")

            payload_raw = env.get("payload")
            if not isinstance(payload_raw, (bytes, bytearray)):
                raise ValueError("infer.payload must be bytes")
            payload = msgpack.unpackb(
                payload_raw, raw=False, object_hook=_decode_np
            )
            actions = session.infer_next_chunk(payload)
            resp_payload = msgpack.packb(
                {"actions": actions}, use_bin_type=True, default=_encode_np
            )
            resp = {
                "ok": True,
                "type": "infer_ack",
                "id": req_id,
                "payload": resp_payload,
            }
            sock.send_multipart([ident, _pack_envelope(resp)])
        except Exception as exc:  # noqa: BLE001
            resp = {
                "ok": False,
                "type": "error",
                "id": req_id,
                "error": {
                    "code": "server_error",
                    "message": str(exc),
                },
            }
            sock.send_multipart([ident, _pack_envelope(resp)])


def main() -> None:
    args = parse_args()
    _serve(args)


if __name__ == "__main__":
    main()


