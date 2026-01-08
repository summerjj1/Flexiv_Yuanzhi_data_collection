"""
Required parameters overview:

Robot side:
    Control:
        - Arm configuration name (built-in or custom)
        - Control interface (ros or non_ros)
        - State interface (same as above)
        - Control mode: joint or eef.
          * joint unit: radian or degree
          * eef representation: homogeneous matrix, position+quaternion,
            position+euler, or lie group
          * default position unit: meter
        - Gripper threshold mode: none / offset / single / dual
        - Action dimension
    Sensing:
        - Camera interfaces (sdk or ros), one per sensor name
        - Other sensor names & interfaces

Model side:
    - Model name (built-in or custom; custom needs normalization data)
    - Client IP
    - Server IP
    - Chunk size
    - Language instruction
    - Inference rate & action rate
    - Non-blocking inference flag
    - Action smoothing mode
    - Observation key mapping for model input
"""

from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Sequence

# TODO: End effector control, multi interface and RTC not implemented yet; reserved for future development.
ControlMode = Literal["joint", "eef"]
JointUnit = Literal["radian", "degree"]
EEFPoseRepresentation = Literal[
    "homogeneous_matrix",
    "position_quaternion",
    "position_euler",
    "lie_group",
]
GripperThresholdMode = Literal["none", "offset", "single", "dual"]
InterfaceKind = Literal["ros", "sdk", "custom"]
TransportKind = Literal["websocket_openpi_client", "zmq"]
TemporalInterpolationMode = Literal[
    "none",
    "exponential_temporal_interpolation",
    "linear_temporal_interpolation",
    "real_time_chunking",
]


@dataclass(frozen=True)
class RobotControlConfig:
    """Robot control and action space description."""

    arm_name: str
    state_interface: InterfaceKind
    controllers: Sequence["ControllerInterfaceConfig"] = field(
        default_factory=tuple
    )
    cameras: Sequence["SensorInterfaceConfig"] = field(default_factory=tuple)
    # Optional initial move command; when provided, will be sent before inference starts.
    init_move_command: Mapping[str, Mapping[str, Any]] | None = None


@dataclass(frozen=True)
class SensorInterfaceConfig:
    """Sensor interface description."""

    name: str
    interface: InterfaceKind
    alias: Sequence[str] = field(default_factory=tuple)
    sensor_name: str | None = None
    obs_type: str | None = None


@dataclass(frozen=True)
class ControllerInterfaceConfig:
    """Controller interface description."""

    controller_name: str
    obs_type: str
    control_interface: InterfaceKind
    control_mode: ControlMode
    joint_unit: JointUnit | None = None
    eef_representation: EEFPoseRepresentation | None = None
    gripper_threshold_mode: GripperThresholdMode = "none"
    # Optional gripper thresholding (applied in PolicyClient.build_move_command after interpolation).
    # This is useful when gripper should be discrete (open/close) even if policy outputs a float.
    gripper_threshold: "GripperThresholdConfig | None" = None
    action_dim: int = 0
    # Optional per-joint angle delta limit; scalar will broadcast to all joints
    angle_delta_limit: Sequence[float] | float | None = None
    enable_angle_delta_limit: bool = False


@dataclass(frozen=True)
class GripperThresholdConfig:
    """Configurable threshold mapping for gripper channels.

    Attributes:
        threshold: Values strictly greater than this are mapped to high_value.
        high_value: Output value when above threshold.
        low_value: Output value when below-or-equal threshold.
        indices: Which action indices to apply the mapping to. If empty, PolicyClient will infer.
    """

    threshold: float = 0.95
    high_value: float = 1.0
    low_value: float = 0.0
    indices: Sequence[int] = field(default_factory=tuple)


@dataclass(frozen=True)
class PolicyWrapperConfig:
    """Mapping for policy I/O fields.

        Attributes:
            entry: Wrapper entry (e.g., "openpiEEFPoseRepresentation = Literal[
        "homogeneous_matrix",
        "position_quaternion",
        "position_euler",
        "lie_group",
    ]
    GripperThresholdMode = Literal["none", "offset", "single", "dual"]
    InterfaceKind = Literal["ros", "sdk", "custom"]").
            observation_keys: Observation key mapping.
            state_key: Key name for state (e.g., "qpos").
            prompt_key: Key name for language instruction.
            controller_state_path: Extraction path for controller state in observation.
                Dot-separated path; "*" wildcard picks first Mapping.
                Examples:
                  - "controllers.*" : single-layer, auto-pick
                  - "controllers.arm.*" : two-layer, under arm
                  - "controllers.lift2_dual_arm" : explicit path
            images_path: Extraction path for images in observation.
                Examples:
                  - "sensors.*.color" : auto-pick sensor color
                  - "sensors.image.*.rgb" : two-layer structure
    """

    entry: str
    observation_keys: Mapping[str, str]
    state_key: str = "qpos"
    prompt_key: str = "prompt"
    # Observation extraction path configuration
    controller_state_path: str = "controllers.*"
    images_path: str = "sensors.*.color"


@dataclass(frozen=True)
class ModelInferenceConfig:
    """Model inference-side configuration."""

    model_name: str
    client_ip: str
    server_ip: str
    chunk_size: int
    language_instruction: str
    inference_rate_hz: float
    action_rate_hz: float
    # Optional: communication/transport type for policy server.
    # If None, PolicyClient will use wrapper default; if wrapper doesn't provide, fallback is "zmq".
    transport: TransportKind | None = None
    non_blocking: bool = False
    interpolation_mode: TemporalInterpolationMode = "none"
    max_non_blocking_inference_hz: float = 10.0
    exponential_interpolation_factor: float = 1.0
    wrapper: PolicyWrapperConfig = field(
        default_factory=lambda: PolicyWrapperConfig(
            entry="", observation_keys={}
        )
    )


@dataclass(frozen=True)
class DeploymentConfig:
    """Unified deployment config covering robot, sensing, and model."""

    name: str
    robot: RobotControlConfig
    model: ModelInferenceConfig


CONFIGS: Sequence[DeploymentConfig] = [
    DeploymentConfig(
        name="lift2_vla_dev",
        robot=RobotControlConfig(
            arm_name="lift2",
            state_interface="ros",
            controllers=(
                ControllerInterfaceConfig(
                    controller_name="lift2_dual_arm",
                    obs_type="qpos",
                    control_interface="ros",
                    control_mode="joint",
                    joint_unit="radian",
                    eef_representation="position_quaternion",
                    gripper_threshold_mode="offset",
                    action_dim=14,
                    enable_angle_delta_limit=True,
                    angle_delta_limit=1
                    * [
                        0.006072,
                        0.018664,
                        0.014556,
                        0.009768,
                        0.007063,
                        0.008024,
                        -1,
                        0.007601,
                        0.022064,
                        0.014512,
                        0.011047,
                        0.013188,
                        0.009549,
                        -1,
                    ],
                ),
            ),
            cameras=(
                SensorInterfaceConfig(
                    name="cam_high",
                    interface="ros",
                    alias=("head", "front"),
                    sensor_name="multirealsense",
                    obs_type="color",
                ),
                SensorInterfaceConfig(
                    name="cam_left_wrist",
                    interface="ros",
                    alias=("left_wrist", "cam_left"),
                    sensor_name="multirealsense",
                    obs_type="color",
                ),
                SensorInterfaceConfig(
                    name="cam_right_wrist",
                    interface="ros",
                    alias=("right_wrist", "cam_right"),
                    sensor_name="multirealsense",
                    obs_type="color",
                ),
            ),
        ),
        model=ModelInferenceConfig(
            model_name="internvl-vla",
            client_ip="0.0.0.0",
            server_ip="127.0.0.1:8000",
            chunk_size=30,
            language_instruction="Bring the shipping label of the package into view, then grasp the package from the conveyor belt and orient the label to myself",
            inference_rate_hz=30.0,
            action_rate_hz=60.0,
            non_blocking=True,
            max_non_blocking_inference_hz=10.0,
            interpolation_mode="linear_temporal_interpolation",
            wrapper=PolicyWrapperConfig(
                entry="openpi",
                observation_keys={
                    "images": "images",
                    "state": "qpos",
                },
            ),
        ),
    ),
    DeploymentConfig(
        name="acone_vla_dev",
        robot=RobotControlConfig(
            arm_name="acone",
            state_interface="ros",
            controllers=(
                ControllerInterfaceConfig(
                    controller_name="acone_dual_arm",
                    obs_type="qpos",
                    control_interface="ros",
                    control_mode="joint",
                    joint_unit="radian",
                    eef_representation="position_quaternion",
                    gripper_threshold_mode="offset",
                    action_dim=14,
                    enable_angle_delta_limit=False,
                ),
            ),
            cameras=(
                SensorInterfaceConfig(
                    name="cam_high",
                    interface="ros",
                    alias=("head", "front"),
                    sensor_name="multirealsense",
                    obs_type="color",
                ),
                SensorInterfaceConfig(
                    name="cam_left_wrist",
                    interface="ros",
                    alias=("left_wrist", "cam_left"),
                    sensor_name="multirealsense",
                    obs_type="color",
                ),
                SensorInterfaceConfig(
                    name="cam_right_wrist",
                    interface="ros",
                    alias=("right_wrist", "cam_right"),
                    sensor_name="multirealsense",
                    obs_type="color",
                ),
            ),
            init_move_command={
                "acone_dual_arm": {
                    "action_type": "joint",
                    "action": [
                        0.0,
                        0.2,
                        0.2,
                        0.0,
                        0.0,
                        0.0,
                        -4.0,
                        0.0,
                        0.2,
                        0.2,
                        0.0,
                        0.0,
                        0.0,
                        -4.0,
                    ],
                }
            },
        ),
        model=ModelInferenceConfig(
            model_name="internvl-vla",
            client_ip="0.0.0.0",
            server_ip="127.0.0.1:8000",
            chunk_size=50,
            language_instruction="clothes new",
            inference_rate_hz=30.0,
            action_rate_hz=50.0,
            non_blocking=True,
            max_non_blocking_inference_hz=10.0,
            interpolation_mode="linear_temporal_interpolation",
            exponential_interpolation_factor=5.0,
            wrapper=PolicyWrapperConfig(
                entry="openpi_acone",
                observation_keys={
                    "images": "images",
                    "state": "qpos",
                },
            ),
        ),
    ),
    DeploymentConfig(
        name="agilex_vla_dev",
        robot=RobotControlConfig(
            arm_name="agilex",
            state_interface="ros",
            controllers=(
                ControllerInterfaceConfig(
                    controller_name="agilex_dual_arm",
                    obs_type="qpos",
                    control_interface="ros",
                    control_mode="joint",
                    joint_unit="radian",
                    eef_representation="position_quaternion",
                    gripper_threshold_mode="offset",
                    action_dim=14,
                    enable_angle_delta_limit=False,
                ),
            ),
            cameras=(
                SensorInterfaceConfig(
                    name="cam_high",
                    interface="ros",
                    alias=("head", "front"),
                    sensor_name="multirealsense",
                    obs_type="color",
                ),
                SensorInterfaceConfig(
                    name="cam_left_wrist",
                    interface="ros",
                    alias=("left_wrist", "cam_left"),
                    sensor_name="multirealsense",
                    obs_type="color",
                ),
                SensorInterfaceConfig(
                    name="cam_right_wrist",
                    interface="ros",
                    alias=("right_wrist", "cam_right"),
                    sensor_name="multirealsense",
                    obs_type="color",
                ),
            ),
        ),
        model=ModelInferenceConfig(
            model_name="internvl-vla",
            client_ip="0.0.0.0",
            server_ip="127.0.0.1:8000",
            chunk_size=50,
            language_instruction="clothes new",
            inference_rate_hz=30.0,
            action_rate_hz=50.0,
            non_blocking=True,
            max_non_blocking_inference_hz=10.0,
            interpolation_mode="linear_temporal_interpolation",
            exponential_interpolation_factor=5.0,
            wrapper=PolicyWrapperConfig(
                entry="openpi",
                observation_keys={
                    "images": "images",
                    "state": "qpos",
                },
            ),
        ),
    ),
    DeploymentConfig(
        name="a2d_vla_dev",
        robot=RobotControlConfig(
            arm_name="a2d",
            state_interface="ros",
            controllers=(
                ControllerInterfaceConfig(
                    controller_name="a2d_dual_arm",
                    # A2D controller state uses dict keys like 'arm_joint_state'
                    obs_type="arm_joint_state",
                    control_interface="ros",
                    control_mode="joint",
                    joint_unit="radian",
                    eef_representation="position_quaternion",
                    gripper_threshold_mode="offset",
                    gripper_threshold=GripperThresholdConfig(
                        threshold=0.95,
                        high_value=1.0,
                        low_value=0.0,
                        indices=(7, 15),
                    ),
                    action_dim=16,
                    enable_angle_delta_limit=False,
                ),
            ),
            cameras=(
                SensorInterfaceConfig(
                    name="cam_high",
                    interface="ros",
                    alias=("/camera/head_color", "head", "front"),
                    sensor_name="a2d_camera",
                    obs_type="color",
                ),
                SensorInterfaceConfig(
                    name="cam_left_wrist",
                    interface="ros",
                    alias=("/camera/hand_left_color", "left_wrist", "cam_left"),
                    sensor_name="a2d_camera",
                    obs_type="color",
                ),
                SensorInterfaceConfig(
                    name="cam_right_wrist",
                    interface="ros",
                    alias=(
                        "/camera/hand_right_color",
                        "right_wrist",
                        "cam_right",
                    ),
                    sensor_name="a2d_camera",
                    obs_type="color",
                ),
            ),
        ),
        model=ModelInferenceConfig(
            model_name="qwena1",
            client_ip="0.0.0.0",
            server_ip="127.0.0.1:8000",
            chunk_size=50,
            language_instruction="Put the pen from the table into the pen holder.",
            inference_rate_hz=30.0,
            action_rate_hz=50.0,
            non_blocking=True,
            max_non_blocking_inference_hz=10.0,
            # interpolation_mode="linear_temporal_interpolation",
            interpolation_mode="exponential_temporal_interpolation",
            exponential_interpolation_factor=5.0,
            wrapper=PolicyWrapperConfig(
                entry="qwena1",
                observation_keys={
                    "images": "images",
                    "state": "qpos",
                },
                prompt_key="task",
            ),
        ),
    ),
]

CONFIG_MAP: Dict[str, DeploymentConfig] = {
    config.name: config for config in CONFIGS
}


def list_configs() -> Sequence[str]:
    """List available config names."""

    return tuple(CONFIG_MAP.keys())


def get_config(config_name: str) -> DeploymentConfig:
    """Fetch deployment config by name."""

    try:
        return CONFIG_MAP[config_name]
    except KeyError as exc:
        raise ValueError(
            f"Config '{config_name}' not found. Available: {', '.join(list_configs())}"
        ) from exc
