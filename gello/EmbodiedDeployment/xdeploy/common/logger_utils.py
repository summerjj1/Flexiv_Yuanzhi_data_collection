import sys
from pathlib import Path
from typing import Optional

from loguru import logger

__all__ = [
    "logger",
    "setup_logger",
    "get_default_format",
    "get_simple_format",
    "get_detailed_format",
    "get_file_format",
]


def get_default_format() -> str:
    """Get default format string with colors."""
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )


def get_simple_format() -> str:
    """Get simple format string with colors."""
    return "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"


def get_detailed_format() -> str:
    """Get detailed format string with colors."""
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<magenta>PID {process}</magenta> | "
        "<level>{message}</level>"
    )


def get_file_format() -> str:
    """Get format string for file output (no colors)."""
    return (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
    colorize: bool = True,
    format_string: Optional[str] = None,
) -> None:
    """
    Setup logger with color formatting and file output.

    Args:
        log_file: Path to log file. If None, only console output.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        rotation: Log rotation size or time (e.g., "10 MB", "1 day").
        retention: Log retention period (e.g., "7 days").
        colorize: Enable color output in console.
        format_string: Custom format string. If None, uses default format.


    Example:
        from xdeploy.common.logger_utils import logger, setup_logger, get_simple_format

        # 使用默认格式
        logger.info("Default format")

        # 使用简单格式
        setup_logger(format_string=get_simple_format())
        logger.info("Simple format")

        # 使用自定义格式
        custom_format = "<level>{level}</level> | {message}"
        setup_logger(format_string=custom_format)
        logger.info("Custom format")
    """
    logger.remove()

    if format_string is None:
        format_string = get_default_format()

    # Console handler with colors
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=colorize,
        backtrace=True,
        diagnose=True,
    )

    # File handler (no colors in file)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format=get_file_format(),
            level=level,
            rotation=rotation,
            retention=retention,
            backtrace=True,
            diagnose=True,
        )


# Initialize with default settings
setup_logger()
