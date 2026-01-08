import time
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar

from xdeploy.common.logger_utils import logger

T = TypeVar("T", bound="Sensor")

# ------------------------------------------------------------------------------
# Lazy import support for sensors to avoid loading all dependencies
# ------------------------------------------------------------------------------
import importlib

# name -> modules list; only import the modules related to the called name
_SENSOR_MODULE_MAP: Dict[str, List[str]] = {
    # RealSense related
    "realsense": ["xdeploy.robot.sensor.camera.realsense"],
    "multirealsense": ["xdeploy.robot.sensor.camera.realsense"],
    "multirealsense_ros": ["xdeploy.robot.sensor.camera.realsense_ros_lift2"],
    "multirealsense_ros2": [
        "xdeploy.robot.sensor.camera.realsense_ros2_acone"
    ],
}

_IMPORTED_SENSOR_MODULES: set[str] = set()


def _lazy_import_for_sensor(sensor_name: str) -> None:
    """Import the modules related to the called name, not all modules."""
    modules = _SENSOR_MODULE_MAP.get(sensor_name, [])
    for module in modules:
        if module in _IMPORTED_SENSOR_MODULES:
            continue
        importlib.import_module(module)
        _IMPORTED_SENSOR_MODULES.add(module)


class SensorMeta(ABCMeta):
    """Metaclass for Sensor to handle Sensor("name") calls."""

    def __call__(cls, *args, **kwargs):
        """Handle Sensor("sensor_name") calls.

        If called on Sensor class itself (not a subclass) with a single string argument
        that matches a registered sensor name, return the registered class.
        Otherwise, proceed with normal instantiation.
        """
        # Only perform sensor lookup if:
        # 1. Called on Sensor class itself (not a subclass)
        # 2. Exactly one positional argument (a string)
        # 3. No keyword arguments
        if (
            cls.__name__ == "Sensor"
            and len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], str)
        ):
            sensor_name = args[0]
            # If registered, return directly
            if sensor_name in cls._registry:
                return cls._registry[sensor_name]

            # If not registered, lazy load the modules related to the called name
            _lazy_import_for_sensor(sensor_name)

            # After lazy loading, check again
            if sensor_name in cls._registry:
                return cls._registry[sensor_name]

            available = (
                ", ".join(cls._registry.keys()) if cls._registry else "none"
            )
            raise KeyError(
                f"Sensor '{sensor_name}' is not registered. "
                f"Available sensors: {available}"
            )

        # Normal instantiation - create instance
        return super().__call__(*args, **kwargs)


class Sensor(ABC, metaclass=SensorMeta):
    """Base class for all sensors.

    This class provides a common interface for all sensor types.
    Subclasses should implement the `_read_data` method to provide
    sensor-specific data reading functionality.

    The class also provides a registry mechanism to register and
    retrieve sensor classes by name.

    Example usage:
        # Register a sensor class (decorator style)
        @Sensor.register("realsense")
        class RealSenseCamera(Camera):
            ...

        # Or register manually
        Sensor.register("realsense", RealSenseCamera)

        # Get a sensor class by name
        CameraClass = Sensor("realsense")
        camera = CameraClass(...)

        # List all registered sensors
        registered = Sensor.list_registered()
        print(registered)  # {'realsense': 'RealSenseCamera', ...}

        # Check if a sensor name is registered
        if Sensor.is_registered("realsense"):
            CameraClass = Sensor("realsense")

    """

    _registry: Dict[str, Type["Sensor"]] = {}

    def __init__(
        self,
        name: Optional[str] = None,
        sensor_type: Optional[str] = None,
        enable_timestamp: bool = True,
    ):
        """Initialize the base sensor.

        Args:
            name: Name of the sensor. If None, uses class name.
            sensor_type: Type of the sensor. If None, uses "sensor".
            enable_timestamp: Whether to include timestamp in readings.
        """
        self.name = name or self.__class__.__name__
        self.sensor_type = sensor_type or "sensor"
        self.enable_timestamp = enable_timestamp
        self._is_initialized = False

        logger.debug(
            f"Initializing sensor: {self.name} (type: {self.sensor_type})"
        )

    @abstractmethod
    def _read_data(self) -> Dict[str, Any]:
        """Read raw data from the sensor.

        This method should be implemented by subclasses to provide
        sensor-specific data reading functionality.

        Returns:
            Dictionary containing sensor data. Keys should be descriptive
            of the data type (e.g., 'color', 'depth', 'imu', etc.).
        """
        pass

    def read(self) -> Optional[Dict[str, Any]]:
        """Read data from the sensor.

        This is the main interface method that should be called to get
        sensor readings. It wraps the `_read_data` method and optionally
        adds timestamp information.

        Returns:
            Dictionary containing sensor data with optional timestamp.
            Returns None if sensor is not initialized or reading fails.
        """
        if not self._is_initialized:
            logger.warning(
                f"Sensor {self.name} is not initialized. Call initialize() first."
            )
            return None

        try:
            data = self._read_data()

            if self.enable_timestamp:

                data["timestamp"] = time.time_ns()

            return data
        except Exception as e:
            logger.error(f"Error reading from sensor {self.name}: {e}")
            return None

    def initialize(self) -> bool:
        """Initialize the sensor.

        This method should be called before using the sensor. Subclasses
        can override this method to perform sensor-specific initialization.

        Returns:
            True if initialization successful, False otherwise.
        """
        self._is_initialized = True
        logger.info(f"Sensor {self.name} initialized successfully")
        return True

    def close(self) -> None:
        """Close the sensor and release resources.

        Subclasses should override this method to properly clean up
        sensor-specific resources.
        """
        self._is_initialized = False
        logger.info(f"Sensor {self.name} closed")

    def is_initialized(self) -> bool:
        """Check if the sensor is initialized.

        Returns:
            True if sensor is initialized, False otherwise.
        """
        return self._is_initialized

    def __repr__(self) -> str:
        """String representation of the sensor."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"type={self.sensor_type}, "
            f"initialized={self._is_initialized})"
        )

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    @classmethod
    def register(
        cls, name: str, sensor_class: Optional[Type[T]] = None
    ) -> Any:
        """Register a sensor class with a given name.

        Can be used as a decorator or as a regular method.

        Args:
            name: Name to register the sensor class under.
            sensor_class: Sensor class to register. If None, returns a decorator.

        Returns:
            If used as decorator, returns the sensor class.
            If used as method, returns None.

        Example:
            # As decorator
            @Sensor.register("realsense")
            class RealSenseCamera(Camera):
                ...

            # As method
            Sensor.register("realsense", RealSenseCamera)
        """
        if sensor_class is None:
            # Used as decorator
            def decorator(sensor_cls: Type[T]) -> Type[T]:
                cls._registry[name] = sensor_cls
                logger.debug(
                    f"Registered sensor class '{sensor_cls.__name__}' as '{name}'"
                )
                return sensor_cls

            return decorator
        else:
            # Used as regular method
            cls._registry[name] = sensor_class
            logger.debug(
                f"Registered sensor class '{sensor_class.__name__}' as '{name}'"
            )
            return sensor_class

    @classmethod
    def get_class(cls, name: str) -> Type["Sensor"]:
        """Get a registered sensor class by name (alternative method).

        This is an alternative to Sensor("name") syntax.

        Args:
            name: Name of the registered sensor class.

        Returns:
            The registered sensor class.

        Raises:
            KeyError: If the sensor name is not registered.

        Example:
            CameraClass = Sensor.get_class("realsense")
            camera = CameraClass(...)
        """
        if name not in cls._registry:
            available = (
                ", ".join(cls._registry.keys()) if cls._registry else "none"
            )
            raise KeyError(
                f"Sensor '{name}' is not registered. "
                f"Available sensors: {available}"
            )
        return cls._registry[name]

    @classmethod
    def list_registered(cls) -> Dict[str, str]:
        """List all registered sensor classes.

        Returns:
            Dictionary mapping sensor names to their class names.
        """
        return {
            name: sensor_class.__name__
            for name, sensor_class in cls._registry.items()
        }

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a sensor name is registered.

        Args:
            name: Name of the sensor to check.

        Returns:
            True if the sensor is registered, False otherwise.
        """
        return name in cls._registry
