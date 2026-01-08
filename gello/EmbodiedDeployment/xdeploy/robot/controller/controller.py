"""
Base controller class for robot controllers.

This module provides BaseController as an abstract base class with registry support
for pluggable controller implementations.
"""

from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, TypeVar

from xdeploy.common.logger_utils import logger

T = TypeVar("T", bound="BaseController")


@dataclass
class BaseControllerConfig:
    """
    Base configuration class for controllers.

    Args:
        name: Name of the controller.
        gripper_type: Type of gripper (optional).
        control_type: Control type, default is "joint".
        extra: Additional configuration parameters as dictionary.
    """

    name: str
    gripper_type: Optional[str] = None
    control_type: str = "joint"
    extra: Dict[str, Any] = field(default_factory=dict)


class ControllerMeta(ABCMeta):
    """Metaclass for BaseController to handle registration."""

    def __call__(cls, *args, **kwargs):
        """Handle BaseController/Controller(\"name\") calls for registry lookup."""
        if (
            cls.__name__ in ("BaseController", "Controller")
            and len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], str)
            and args[0] in cls._registry
        ):
            return cls._registry[args[0]]
        elif (
            cls.__name__ in ("BaseController", "Controller")
            and len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], str)
            and args[0] not in cls._registry
        ):
            available = (
                ", ".join(cls._registry.keys()) if cls._registry else "none"
            )
            raise KeyError(
                f"Controller '{args[0]}' is not registered. Available controllers: {available}"
            )

        return super().__call__(*args, **kwargs)


class BaseController(ABC, metaclass=ControllerMeta):
    """
    Base class for all robot controllers with registry support.

    This is an abstract base class that provides:
    - Registry mechanism for controller registration
    - Common interface for controllers (set_up, reset, get_state)
    - Configuration-based initialization

    Example:
        @BaseController.register("mock_controller")
        class MockController(BaseController):
            def __init__(self, config: BaseControllerConfig):
                super().__init__(config)

            ...
    """

    _registry: Dict[str, Type["BaseController"]] = {}

    @classmethod
    def register(
        cls, name: str, controller_class: Optional[Type[T]] = None
    ) -> Any:
        """Register a controller class with a given name.

        Can be used as a decorator or as a regular method.

        Args:
            name: Name to register the controller class under.
            controller_class: Controller class to register. If None, returns a decorator.

        Returns:
            If used as decorator, returns the controller class.
            If used as method, returns None.

        Example:
            # As decorator
            @BaseController.register("mock_controller")
            class MockController(BaseController):
                ...

            # As method
            BaseController.register("mock_controller", MockController)
        """
        if controller_class is None:
            # Used as decorator
            def decorator(controller_cls: Type[T]) -> Type[T]:
                cls._registry[name] = controller_cls
                logger.debug(
                    f"Registered controller class '{controller_cls.__name__}' as '{name}'"
                )
                return controller_cls

            return decorator
        else:
            # Used as regular method
            cls._registry[name] = controller_class
            logger.debug(
                f"Registered controller class '{controller_class.__name__}' as '{name}'"
            )
            return controller_class

    @classmethod
    def get_class(cls, name: str) -> Type["BaseController"]:
        """Get a registered controller class by name.

        Args:
            name: Name of the registered controller class.

        Returns:
            The registered controller class.

        Raises:
            KeyError: If the controller name is not registered.

        Example:
            ControllerClass = BaseController.get_class("mock_controller")
            config = BaseControllerConfig(name="arm")
            controller = ControllerClass(config)
        """
        if name not in cls._registry:
            available = (
                ", ".join(cls._registry.keys()) if cls._registry else "none"
            )
            raise KeyError(
                f"Controller '{name}' is not registered. Available controllers: {available}"
            )
        return cls._registry[name]

    @classmethod
    def list_registered(cls) -> Dict[str, str]:
        """List all registered controller classes.

        Returns:
            Dictionary mapping controller names to their class names.
        """
        return {
            name: controller_class.__name__
            for name, controller_class in cls._registry.items()
        }

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a controller name is registered.

        Args:
            name: Name of the controller to check.

        Returns:
            True if the controller is registered, False otherwise.
        """
        return name in cls._registry

    def __init__(self, config: BaseControllerConfig):
        """
        Initialize the controller.

        Args:
            config: Controller configuration object.
        """
        self.config = config

    @abstractmethod
    def set_up(self) -> None:
        """Initialize the controller. Should be called before using the controller."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the controller to initial state."""
        pass

    @abstractmethod
    def get_state(self) -> Any:
        """Get current controller state (e.g., joint positions + gripper width).

        Returns:
            Controller state as numpy array or dict.
        """
        pass

    @abstractmethod
    def apply_action(
        self, action: Any, action_type: Optional[str] = None
    ) -> None:
        """Apply action to the controller.

        Args:
            action: Action to apply (list or array).
            action_type: Type of action. If None, behavior is implementation-dependent
                (may use config.control_type or default to "joint").
        """
        pass

    @abstractmethod
    def help(self) -> str:
        """Return help message of the controller."""
        pass

    def __repr__(self) -> str:
        """Return string representation of the controller."""
        return f"{self.__class__.__name__}(config={self.config})"


class Controller(BaseController):
    """
    Convenience alias for BaseController when using registry lookup.

    Example:
        ControllerClass = Controller("mock_controller")
        config = BaseControllerConfig(name="arm")
        controller = ControllerClass(config)
    """

    pass
