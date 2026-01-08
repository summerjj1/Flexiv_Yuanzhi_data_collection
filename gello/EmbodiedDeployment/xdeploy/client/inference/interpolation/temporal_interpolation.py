from __future__ import annotations
import logging
from typing import Optional

import numpy as np

from config.config import DeploymentConfig
from xdeploy.common.logger_utils import logger


class TemporalInterpolator:
    """Apply temporal smoothing to action chunks returned by a policy.

    Args:
        config: DeploymentConfig used to parse control and interpolation modes.
        smoothing_factor: Default smoothing weight in [0, 1]; inferred from
            config.model.interpolation_mode when None.
    """

    def __init__(
        self, config: DeploymentConfig, smoothing_factor: float | None = None
    ) -> None:
        self._config = config
        # Use controller-level config if available
        ctrl_cfg = None
        controllers = getattr(config.robot, "controllers", None)
        if controllers:
            ctrl_cfg = controllers[0]

        self._control_mode = getattr(
            ctrl_cfg,
            "control_mode",
            getattr(config.robot, "control_mode", "joint"),
        )
        self._eef_representation = getattr(
            ctrl_cfg,
            "eef_representation",
            getattr(config.robot, "eef_representation", None),
        )
        self._chunk_size = getattr(config.model, "chunk_size", 1)
        self._action_dim = getattr(ctrl_cfg, "action_dim", None)
        if self._action_dim is None:
            self._action_dim = getattr(config.robot, "action_dim", 0) or None
        self._smoothing_factor = (
            self._infer_default_factor(config)
            if smoothing_factor is None
            else smoothing_factor
        )
        self._orientation_slice = self._infer_orientation_slice()
        self._action_buffer: Optional[np.ndarray] = None
        self._interpolation_mode = getattr(
            config.model, "interpolation_mode", "none"
        )
        default_min = max(1, self._chunk_size // 8) * 2
        default_max = max(default_min, self._chunk_size // 8) * 6
        self._min_interpolate_actions = int(
            max(
                1,
                getattr(
                    config.model, "interpolation_min_actions", default_min
                ),
            )
        )
        self._max_interpolate_actions = int(
            max(
                self._min_interpolate_actions,
                getattr(
                    config.model, "interpolation_max_actions", default_max
                ),
            )
        )
        self._executed_steps = 0

    @property
    def action_buffer(self) -> Optional[np.ndarray]:
        """Return the current cached action chunk."""

        if self._action_buffer is None:
            return None
        return self._action_buffer.copy()

    def buffer_size(self) -> int:
        if self._action_buffer is None:
            return 0
        return int(self._action_buffer.shape[0])

    def peek_action(self) -> Optional[np.ndarray]:
        """Peek the first pending action without removing it."""

        if self._action_buffer is None or self._action_buffer.size == 0:
            return None
        return self._action_buffer[0].copy()

    def pop_action(self) -> Optional[np.ndarray]:
        """Pop the leftmost action; keep the last one in the buffer."""

        action = self.peek_action()
        if action is None:
            return None
        if (
            self._action_buffer is not None
            and self._action_buffer.shape[0] > 1
        ):
            self._action_buffer = self._action_buffer[1:, :]
        return action

    def mark_executed_steps(self, step_count: int = 1) -> None:
        """Record executed action steps for latency trimming."""

        if step_count <= 0:
            return
        self._executed_steps += int(step_count)

    def reset(self) -> None:
        """Clear the action buffer."""

        self._action_buffer = None
        self._executed_steps = 0

    def update(self, action_chunk: np.ndarray) -> np.ndarray:
        """Insert a new action chunk and smooth with latency/length constraints.

        Args:
            action_chunk: Action chunk from the policy ([chunk, action_dim]).

        Returns:
            Updated action buffer (np.ndarray).
        """

        chunk = np.asarray(action_chunk, dtype=np.float32)
        chunk = np.atleast_2d(chunk)
        if chunk.size == 0:
            raise ValueError("action_chunk cannot be empty")

        latency_steps = max(0, self._drain_executed_steps())
        trimmed = self._trim_latency(chunk, latency_steps)
        if trimmed.size == 0:
            logging.warning(
                "TemporalInterpolator: chunk length (%d) <= latency (%d); clearing "
                "buffer and waiting for new data",
                chunk.shape[0],
                latency_steps,
            )
            self._action_buffer = None
            return np.empty((0, chunk.shape[1]), dtype=np.float32)

        if self._action_buffer is None or self._action_buffer.size == 0:
            self._action_buffer = trimmed.copy()
            return self._action_buffer.copy()
        # logger.info(f"buffer:\n {self._action_buffer[:,-1]}\n and trimmed:\n {trimmed[:,-1]}")
        updated = self._interpolate_with_buffer(trimmed)
        self._action_buffer = updated

        return self._action_buffer.copy()

    def _infer_default_factor(self, config: DeploymentConfig) -> float:
        mode = getattr(config.model, "interpolation_mode", "none")
        lookup = {
            "exponential_temporal_interpolation": 0.25,
            "linear_temporal_interpolation": 0.5,
            "real_time_chunking": 0.75,
            "none": 1.0,
        }
        value = lookup.get(mode, 1.0)
        return float(np.clip(value, 0.0, 1.0))

    def _infer_orientation_slice(self) -> Optional[slice]:
        if self._control_mode not in ("eef", "ee"):
            return None

        representation = self._eef_representation
        if representation == "position_quaternion":
            return slice(3, 7)
        if representation == "position_euler":
            return slice(3, 6)
        return None

    def _slerp_quaternion_batch(
        self, old: np.ndarray, new: np.ndarray, weight: np.ndarray
    ) -> np.ndarray:
        blended = np.empty_like(old)
        for idx in range(old.shape[0]):
            blended[idx] = self._slerp_quaternion(
                old[idx], new[idx], float(weight[idx])
            )
        return blended

    @staticmethod
    def _slerp_quaternion(
        q0: np.ndarray, q1: np.ndarray, t: float
    ) -> np.ndarray:
        """Perform single-step quaternion SLERP while avoiding divergence."""

        q0 = np.asarray(q0, dtype=np.float32)
        q1 = np.asarray(q1, dtype=np.float32)
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)

        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * np.clip(t, 0.0, 1.0)
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        result = s0 * q0 + s1 * q1
        return result / np.linalg.norm(result)

    def _trim_latency(
        self, chunk: np.ndarray, latency_steps: int
    ) -> np.ndarray:
        latency = max(0, int(latency_steps))
        if latency <= 0:
            return chunk
        if latency >= chunk.shape[0]:
            return np.empty((0, chunk.shape[1]), dtype=np.float32)
        return chunk[latency:, :]

    def _drain_executed_steps(self) -> int:
        latency = self._executed_steps
        self._executed_steps = 0
        return latency

    def _interpolate_with_buffer(self, chunk: np.ndarray) -> np.ndarray:
        assert self._action_buffer is not None
        buffer_len = self._action_buffer.shape[0]
        chunk_len = chunk.shape[0]
        overlap = min(buffer_len, chunk_len)
        if overlap <= 0:
            return chunk.copy()

        window_len = self._constrain_window_length(overlap)
        old_block = self._slice_with_padding(self._action_buffer, window_len)
        new_block = self._slice_with_padding(chunk, window_len)
        weights = self._build_transition_weights(window_len)
        blended = self._mix_blocks(old_block, new_block, weights)
        tail_start = min(chunk_len, window_len)
        if chunk_len > tail_start:
            blended = np.vstack((blended, chunk[tail_start:, :]))
        return blended

    def _constrain_window_length(self, overlap: int) -> int:
        length = overlap
        length = min(length, self._max_interpolate_actions)
        length = max(length, self._min_interpolate_actions)
        return int(length)

    @staticmethod
    def _slice_with_padding(data: np.ndarray, target_len: int) -> np.ndarray:
        if data.size == 0:
            raise ValueError(
                "Cannot build interpolation window from an empty array"
            )
        current = data.shape[0]
        if current >= target_len:
            return data[:target_len, :].copy()
        pad_count = target_len - current
        pad_block = np.repeat(data[-1:, :], pad_count, axis=0)
        return np.vstack((data, pad_block))

    def _build_transition_weights(self, length: int) -> np.ndarray:
        if length <= 0:
            return np.zeros((0, 1), dtype=np.float32)
        if length <= 0:
            return np.zeros((0, 1), dtype=np.float32)

        mode = self._interpolation_mode
        if length == 1:
            base = np.array([1.0], dtype=np.float32)
        else:
            progress = np.linspace(0.0, 1.0, num=length, dtype=np.float32)
            if mode == "exponential_temporal_interpolation":
                beta = max(self._smoothing_factor, 1e-3)
                # Read exponential interpolation factor from config; default to 1.0
                exp_factor = getattr(
                    self._config.model, "exponential_interpolation_factor", 1.0
                )
                # Use a configurable multiplier to accelerate early transition
                # Push weights to switch to the new chunk early instead of late
                base = 1.0 - np.exp(-beta * progress * exp_factor)
                max_val = base[-1] if base[-1] > 0 else 1.0
                base = base / max_val
            elif mode == "linear_temporal_interpolation":
                base = progress
            elif mode == "real_time_chunking":
                base = np.sqrt(progress)
            elif mode == "none":
                base = np.ones(length, dtype=np.float32)
            else:
                base = progress

        if mode != "none" and length > 1:
            base[0] = 0.0
            base[-1] = 1.0
        return self._clip_alpha(base).reshape(length, 1)

    @staticmethod
    def _clip_alpha(alpha: np.ndarray) -> np.ndarray:
        return np.clip(alpha, 0.0, 1.0)

    def _mix_blocks(
        self, old_block: np.ndarray, new_block: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        blended = old_block + weights * (new_block - old_block)
        if (
            self._control_mode in ("eef", "ee")
            and self._orientation_slice is not None
        ):
            blended[:, self._orientation_slice] = self._blend_orientation(
                old_block[:, self._orientation_slice],
                new_block[:, self._orientation_slice],
                weights[:, 0],
            )
        return blended

    def _blend_orientation(
        self, old: np.ndarray, new: np.ndarray, weight: np.ndarray
    ) -> np.ndarray:
        """Custom smoothing for the EEF orientation portion."""

        if old.shape[1] == 4:
            return self._slerp_quaternion_batch(old, new, weight)
        return old + weight[:, None] * (new - old)
