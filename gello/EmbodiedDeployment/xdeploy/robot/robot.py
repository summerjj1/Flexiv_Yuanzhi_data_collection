"""
Base robot class for managing controllers and sensors.

This module provides an abstract Robot class that can be inherited to create
robot implementations with pluggable controllers and sensors.
"""

import pickle
from abc import ABC, abstractmethod
from shutil import ExecError
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from xdeploy.common.logger_utils import logger

# Import base classes
try:
    from xdeploy.robot.controller import BaseController
    from xdeploy.robot.sensor.sensor import Sensor
except ImportError as err:
    # Fallback for type checking
    Sensor = Any
    BaseController = Any
    logger.error(f"Failed to import controller or sensor: {err}")


class ControllerAction(BaseModel):
    """Action data for a single controller.

    Attributes:
        action: Either a numeric list/tuple or a dict payload depending on controller.
        action_type: Type of action, default is "joint".
    """

    action: Union[List[Union[int, float]], Dict[str, Any]] = Field(
        ..., description="Action payload (list of numbers or dict)"
    )
    action_type: str = Field(default="joint", description="Type of action")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v):
        """Validate that action is a numeric list/tuple or non-empty dict."""
        if isinstance(v, tuple):
            v = list(v)

        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("action list must not be empty")
            if not all(isinstance(item, (int, float)) for item in v):
                raise TypeError("action list must contain numbers")
            return v

        if isinstance(v, dict):
            if not v:
                raise ValueError("action dict must not be empty")
            return v

        raise TypeError(
            "action must be a list/tuple of numbers or a dict payload"
        )

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v):
        """Validate action_type."""
        if not isinstance(v, str):
            raise ValueError("action_type must be a string")
        return v


class MoveData:
    """Move data structure for robot control with validation.

    This class provides a dictionary-like interface with Pydantic validation
    for controller actions. Keys are controller names, values are ControllerAction.

    Example:
        # From dictionary (validated)
        data = {
            "left_arm": {
                "action": [0.1, 0.2, 0.3],
                "action_type": "joint"
            }
        }
        move_data = MoveData.from_dict(data)

        # Access as dict
        for controller_name, action in move_data.items():
            print(f"{controller_name}: {action['action']}")
    """

    def __init__(self, data: Optional[Dict[str, ControllerAction]] = None):
        """Initialize MoveData with validated controller actions.

        Args:
            data: Dictionary mapping controller names to ControllerAction instances.
        """
        self._data: Dict[str, ControllerAction] = data or {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MoveData":
        """Create MoveData from a dictionary with validation.

        Args:
            data: Dictionary with controller names as keys and action dicts as values.

        Returns:
            MoveData instance with validated data.

        Raises:
            ValueError: If validation fails for any controller action.

        Example:
            data = {
                "left_arm": {
                    "action": [0.1, 0.2, 0.3],
                    "action_type": "joint"
                }
            }
            move_data = MoveData.from_dict(data)
        """
        validated_data = {}
        for controller_name, action_data in data.items():
            if isinstance(action_data, dict):
                try:
                    validated_data[controller_name] = ControllerAction(
                        **action_data
                    )
                except Exception as e:
                    raise ValueError(
                        f"Invalid action data for controller '{controller_name}': {e}"
                    ) from e
            elif isinstance(action_data, ControllerAction):
                validated_data[controller_name] = action_data
            else:
                raise ValueError(
                    f"Invalid action data for controller '{controller_name}': "
                    f"expected dict or ControllerAction, got {type(action_data).__name__}"
                )
        return cls(validated_data)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert MoveData to dictionary format.

        Returns:
            Dictionary representation of MoveData.
        """
        return {
            controller_name: controller_action.model_dump()
            for controller_name, controller_action in self._data.items()
        }

    def items(self):
        """Return items iterator for dict-like access."""
        return self.to_dict().items()

    def keys(self):
        """Return keys iterator for dict-like access."""
        return self._data.keys()

    def values(self):
        """Return values iterator for dict-like access."""
        return self._data.values()

    def __getitem__(self, key: str) -> ControllerAction:
        """Get controller action by name."""
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        """Check if controller name exists."""
        return key in self._data

    def __len__(self) -> int:
        """Return number of controllers."""
        return len(self._data)


class Robot(ABC):
    """
    Abstract base robot class for managing controllers and sensors.

    Supports configuration-based initialization using registered controllers and sensors.

    Example:
        # Using config-based initialization
        config = {
            "controllers": {
                "left_arm": {
                    "controller_type": "mock_controller",
                    "args": {"name": "left_arm", "num_joints": 7}
                }
            },
            "sensors": {
                "head_camera": {
                    "sensor_type": "mock_camera",
                    "args": {"name": "head_camera"}
                }
            }
        }
        robot = MyRobot(config=config)

        # Or manual initialization
        class MyRobot(Robot):
            def __init__(self):
                super().__init__()
                self.controllers = {"left_arm": MyArmController()}
                self.sensors = {"camera": MyCamera()}

            def set_up(self):
                # Initialize hardware
                pass
    """

    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the robot.

        Args:
            name: Name of the robot. If None, uses class name.
            config: Optional configuration dictionary for initializing controllers and sensors.
                Format:
                {
                    "controllers": {
                        "controller_name": {
                            "controller_type": "registered_controller_name",
                            "args": {...}  # Arguments for controller initialization
                        }
                    },
                    "sensors": {
                        "sensor_name": {
                            "sensor_type": "registered_sensor_name",
                            "args": {...}  # Arguments for sensor initialization
                        }
                    }
                }
        """
        self.name = name or self.__class__.__name__
        self.controllers: Dict[str, Any] = {}
        self.sensors: Dict[str, Any] = {}
        self._is_setup = False

        # Initialize from config if provided
        if config:
            self._initialize_from_config(config)

    def _initialize_from_config(self, config: Dict[str, Any]) -> None:
        """
        Initialize controllers and sensors from configuration dictionary.

        Args:
            config: Configuration dictionary with controllers and sensors setup.
        """
        # Initialize controllers from config
        if "controllers" in config:
            self._initialize_controllers_from_config(config["controllers"])

        # Initialize sensors from config using registry
        if "sensors" in config:
            self._initialize_sensors_from_config(config["sensors"])

    def _initialize_controllers_from_config(
        self, controllers_config: Dict[str, Any]
    ) -> None:
        """
        Initialize controllers from configuration.

        Args:
            controllers_config: Controllers configuration dictionary.
                Format: {"controller_name": {"controller_type": "...", "args": {...}}}
        """
        for controller_name, controller_config in controllers_config.items():
            controller_type = controller_config.get("controller_type")
            controller_args = controller_config.get("args", {})

            if not controller_type:
                continue

            # Try to get controller class from BaseController registry if available
            if isinstance(controller_type, str):
                # Try to get from registry if BaseController has registry
                controller_class = self._get_controller_class(controller_type)
                if controller_class is None:
                    continue
            else:
                controller_class = controller_type

            try:
                controller = controller_class(**controller_args)
                self.controllers[controller_name] = controller
            except Exception as e:
                from xdeploy.common.logger_utils import logger

                logger.error(
                    f"Failed to initialize controller {controller_name}: {e}"
                )

    def _get_controller_class(self, controller_type: str):
        """
        Get controller class from BaseController registry.

        Args:
            controller_type: Name of the controller type (registered name).

        Returns:
            Controller class or None if not found.
        """
        try:
            if BaseController.is_registered(controller_type):
                return BaseController.get_class(controller_type)
        except Exception:
            pass
        return None

    def _initialize_sensors_from_config(
        self, sensors_config: Dict[str, Any]
    ) -> None:
        """
        Initialize sensors from configuration using Sensor registry.

        Args:
            sensors_config: Sensors configuration dictionary.
                Format: {"sensor_name": {"sensor_type": "...", "args": {...}}}
        """
        for sensor_name, sensor_config in sensors_config.items():
            sensor_type = sensor_config.get("sensor_type")
            sensor_args = sensor_config.get("args", {})

            if not sensor_type:
                continue

            # Get sensor class from Sensor registry
            try:
                if not Sensor.is_registered(sensor_type):
                    from xdeploy.common.logger_utils import logger

                    logger.warning(
                        f"Sensor '{sensor_type}' not found in registry. "
                        f"Available: {Sensor.list_registered()}. "
                        f"Make sure the sensor class is imported before initialization."
                    )
                    continue

                sensor_class = Sensor.get_class(sensor_type)
                # Merge sensor_name into args
                sensor_init_args = {"name": sensor_name, **sensor_args}
                sensor = sensor_class(**sensor_init_args)
                self.sensors[sensor_name] = sensor
            except Exception as e:
                from xdeploy.common.logger_utils import logger

                logger.error(f"Failed to initialize sensor {sensor_name}: {e}")

    def set_up(self) -> None:
        """
        Set up the robot by initializing all controllers and sensors.

        This method should be called after instantiating the robot and
        before using any robot functionality.

        Subclasses can override this method to add custom initialization logic,
        but should call super().set_up() to initialize controllers and sensors.
        """
        # Initialize controllers
        for controller_name, controller in self.controllers.items():
            if hasattr(controller, "set_up"):
                controller.set_up()
            elif hasattr(controller, "initialize"):
                controller.initialize()

        # Initialize sensors
        for sensor_name, sensor in self.sensors.items():
            if hasattr(sensor, "initialize"):
                sensor.initialize()

        self._is_setup = True
        logger.info(f"Robot {self.name} is ready")

    def get(self) -> Dict[str, Any]:
        """
        Get current observation from all controllers and sensors.

        Returns:
            Dictionary containing observations. Structure is implementation-dependent.
        """
        controller_data = {}
        sensor_data = {}

        # Get controller observations
        for controller_name, controller in self.controllers.items():
            if hasattr(controller, "get_state"):
                controller_data[controller_name] = controller.get_state()
            else:
                raise ValueError(
                    f"Controller {controller_name} does not have get_state method"
                )
        # Get sensor readings
        for sensor_name, sensor in self.sensors.items():
            if hasattr(sensor, "read"):
                reading = sensor.read()
                if reading is not None:
                    sensor_data[sensor_name] = reading
            else:
                raise ValueError(
                    f"Sensor {sensor_name} does not have read method"
                )

        return {"controllers": controller_data, "sensors": sensor_data}

    def move(self, move_data: Union[Dict[str, Any], MoveData]) -> None:
        """
        Move the robot according to the provided action data.

        Args:
            move_data: Action data structure. Can be:
                - MoveData instance (recommended)
                - Dictionary format:
                  {
                      "controller_name": {
                          "action": [list of action values] or {"target_pose": ...},
                          "action_type": "joint"  # optional, default is "joint"
                      }
                  }

        Example:
            # Using MoveData model (recommended)
            from xdeploy.robot.robot import MoveData, ControllerAction
            move_data = MoveData(
                left_arm=ControllerAction(action=[0.1, 0.2, 0.3], action_type="joint")
            )
            robot.move(move_data)

            # Using dictionary (also supported)
            move_data = {
                "left_arm": {
                    "action": [0.1, 0.2, 0.3],
                    "action_type": "joint"
                }
            }
            robot.move(move_data)
        """
        # Convert dict to MoveData if needed
        # Validate input data format
        if isinstance(move_data, dict):
            try:
                move_data = MoveData.from_dict(move_data)
            except Exception as e:
                raise ValueError(f"Invalid move_data format: {e}") from e
        elif not isinstance(move_data, MoveData):
            raise TypeError(
                f"move_data must be MoveData or dict, got {type(move_data).__name__}"
            )

        # Process each controller action (move_data is now guaranteed to be MoveData)
        move_data_dict = move_data.to_dict()
        for controller_name, controller_action in move_data_dict.items():
            if controller_name not in self.controllers:
                logger.warning(
                    f"Controller '{controller_name}' not found. Available: {list(self.controllers.keys())}"
                )
                continue

            controller = self.controllers[controller_name]

            # Extract action and action_type from validated data
            action = controller_action["action"]
            action_type = controller_action.get("action_type", "joint")

            if hasattr(controller, "apply_action"):
                controller.apply_action(action, action_type=action_type)
            else:
                raise AttributeError(
                    f"Controller {controller_name} does not have apply_action method"
                )

    def reset(self) -> None:
        """
        Reset the robot to initial state.

        Subclasses should override this method to implement robot-specific
        reset logic (e.g., moving to home position).
        """
        for controller_name, controller in self.controllers.items():
            if hasattr(controller, "reset"):
                controller.reset()

    def is_setup(self) -> bool:
        """
        Check if the robot is set up.

        Returns:
            True if the robot has been set up, False otherwise.
        """
        return self._is_setup

    def is_start(self) -> bool:
        """
        Check if the robot is started (same as is_setup).

        Returns:
            True if the robot has been set up, False otherwise.
        """
        return self._is_setup

    def close(self) -> None:
        """
        Close all controllers and sensors, releasing resources.

        Subclasses should override this method to properly clean up resources.
        """
        for controller_name, controller in self.controllers.items():
            if hasattr(controller, "close"):
                controller.close()

        # Close sensors
        for sensor_name, sensor in self.sensors.items():
            if hasattr(sensor, "close"):
                sensor.close()

        self._is_setup = False

    def __enter__(self):
        """Context manager entry."""
        self.set_up()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation of the robot."""
        return f"{self.__class__.__name__}(name={self.name})"
