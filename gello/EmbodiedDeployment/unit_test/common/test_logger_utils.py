"""
Unit tests for logger_utils module.
"""
import os
import tempfile
from pathlib import Path

import pytest

from xdeploy.common.logger_utils import (
    get_default_format,
    get_detailed_format,
    get_file_format,
    get_simple_format,
    logger,
    setup_logger,
)


class TestFormatFunctions:
    """Test format string functions."""

    def test_get_default_format(self):
        """Test default format string."""
        format_str = get_default_format()
        assert isinstance(format_str, str)
        assert "{time" in format_str
        assert "{level" in format_str
        assert "{name" in format_str
        assert "{function" in format_str
        assert "{line" in format_str
        assert "{message" in format_str
        assert "<green>" in format_str
        assert "<cyan>" in format_str
        assert "<level>" in format_str

    def test_get_simple_format(self):
        """Test simple format string."""
        format_str = get_simple_format()
        assert isinstance(format_str, str)
        assert "{time" in format_str
        assert "{level" in format_str
        assert "{message" in format_str
        assert "<green>" in format_str
        assert "<level>" in format_str
        # Should not contain detailed info
        assert "{name}" not in format_str
        assert "{function}" not in format_str

    def test_get_detailed_format(self):
        """Test detailed format string."""
        format_str = get_detailed_format()
        assert isinstance(format_str, str)
        assert "{time" in format_str
        assert "{level" in format_str
        assert "{name" in format_str
        assert "{function" in format_str
        assert "{line" in format_str
        assert "{process" in format_str
        assert "{message" in format_str
        assert "<magenta>" in format_str

    def test_get_file_format(self):
        """Test file format string (no colors)."""
        format_str = get_file_format()
        assert isinstance(format_str, str)
        assert "{time" in format_str
        assert "{level" in format_str
        assert "{name" in format_str
        assert "{function" in format_str
        assert "{line" in format_str
        assert "{message" in format_str
        # Should not contain color tags
        assert "<green>" not in format_str
        assert "<cyan>" not in format_str
        assert "<level>" not in format_str
        assert "<magenta>" not in format_str


class TestSetupLogger:
    """Test setup_logger function."""

    def test_setup_logger_default(self):
        """Test setup_logger with default parameters."""
        setup_logger()
        # Logger should be configured
        assert logger is not None

    def test_setup_logger_custom_level(self):
        """Test setup_logger with custom level."""
        setup_logger(level="DEBUG")
        logger.debug("Test debug message")
        # Should not raise exception

    def test_setup_logger_custom_format(self):
        """Test setup_logger with custom format string."""
        custom_format = "<level>{level}</level> | {message}"
        setup_logger(format_string=custom_format)
        logger.info("Test message")
        # Should not raise exception

    def test_setup_logger_with_file(self):
        """Test setup_logger with file output."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".log"
        ) as f:
            log_file = f.name

        try:
            setup_logger(log_file=log_file, level="INFO")
            logger.info("Test log message")

            # Check if file was created
            assert Path(log_file).exists()

            # Check if log content was written
            with open(log_file, "r") as f:
                content = f.read()
                assert "Test log message" in content
                assert "INFO" in content
        finally:
            # Cleanup
            if Path(log_file).exists():
                os.unlink(log_file)

    def test_setup_logger_file_directory_creation(self):
        """Test that setup_logger creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "subdir", "test.log")
            setup_logger(log_file=log_file, level="INFO")
            logger.info("Test message")

            # Check if directory and file were created
            assert Path(log_file).exists()
            assert Path(log_file).parent.exists()

    def test_setup_logger_no_colorize(self):
        """Test setup_logger with colorize disabled."""
        setup_logger(colorize=False)
        logger.info("Test message")
        # Should not raise exception

    def test_setup_logger_with_format_functions(self):
        """Test setup_logger with different format functions."""
        formats = [
            get_default_format(),
            get_simple_format(),
            get_detailed_format(),
        ]

        for fmt in formats:
            setup_logger(format_string=fmt)
            logger.info("Test message")
            # Should not raise exception


class TestLoggerUsage:
    """Test logger usage after setup."""

    def test_logger_info(self):
        """Test logger.info()."""
        setup_logger(level="INFO")
        logger.info("Info message")
        # Should not raise exception

    def test_logger_debug(self):
        """Test logger.debug()."""
        setup_logger(level="DEBUG")
        logger.debug("Debug message")
        # Should not raise exception

    def test_logger_warning(self):
        """Test logger.warning()."""
        setup_logger(level="WARNING")
        logger.warning("Warning message")
        # Should not raise exception

    def test_logger_error(self):
        """Test logger.error()."""
        setup_logger(level="ERROR")
        logger.error("Error message")
        # Should not raise exception

    def test_logger_success(self):
        """Test logger.success()."""
        setup_logger(level="INFO")
        logger.success("Success message")
        # Should not raise exception

    def test_logger_exception(self):
        """Test logger.exception()."""
        setup_logger(level="ERROR")
        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("Exception occurred")
        # Should not raise exception

    def test_logger_with_file_output(self):
        """Test logger writes to file correctly."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".log"
        ) as f:
            log_file = f.name

        try:
            setup_logger(log_file=log_file, level="INFO")
            logger.info("File test message")
            logger.warning("File warning message")
            logger.error("File error message")

            # Read file and verify content
            with open(log_file, "r") as f:
                content = f.read()
                assert "File test message" in content
                assert "File warning message" in content
                assert "File error message" in content
                assert "INFO" in content
                assert "WARNING" in content
                assert "ERROR" in content
        finally:
            if Path(log_file).exists():
                os.unlink(log_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
