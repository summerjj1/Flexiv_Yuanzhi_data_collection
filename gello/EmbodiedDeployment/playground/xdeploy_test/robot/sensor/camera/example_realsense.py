"""Debug script for RealSense camera functionality.

This script demonstrates various usage patterns of RealSense cameras,
including single camera, multi-camera, blocking and non-blocking modes.
"""

import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image

from xdeploy.common.logger_utils import logger
from xdeploy.robot.sensor import Sensor
from xdeploy.robot.sensor.camera import NonBlockingCameraWrapper

try:
    from xdeploy.robot.sensor.camera.realsense import (
        REALSENSE_CAM_MAP,
        get_available_cameras,
    )
except ImportError as e:
    print(f"Warning: Could not import RealSense functions: {e}")
    get_available_cameras = lambda: {}


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def example_available_cameras():
    """Test: List all available cameras."""
    print_section("1. Testing Available Cameras")

    try:
        available = get_available_cameras()
        if available:
            print(f"Found {len(available)} camera(s):")
            for serial, name in available.items():
                print(f"  - Serial: {serial}, Name: {name}")
            return list(available.keys())
        else:
            print("No RealSense cameras found!")
            return []
    except Exception as e:
        print(f"Error getting available cameras: {e}")
        return []


def example_single_camera_blocking(serial_number: Optional[str] = None):
    """Test: Single camera in blocking mode."""
    print_section("2. Testing Single Camera (Blocking Mode)")
    save_dir = "tmp"

    try:
        CameraClass = Sensor("realsense")

        # Use first available camera if serial not provided
        if serial_number is None:
            available = get_available_cameras()
            if not available:
                print("No cameras available for testing")
                return False
            serial_number = list(available.keys())[0]
            print(f"Using first available camera: {serial_number}")

        print(f"Creating camera with serial: {serial_number}")
        cam_info = REALSENSE_CAM_MAP["D455"]["640x480_30"]
        camera = CameraClass(
            serial_number=serial_number,
            image_width=cam_info["image_width"],
            image_height=cam_info["image_height"],
            fps=cam_info["fps"],
            enable_depth=True,
        )

        print("Initializing camera...")
        if not camera.initialize():
            print("Failed to initialize camera")
            return False

        print("Camera initialized successfully")
        print(f"  - Name: {camera.name}")
        print(f"  - Serial: {camera.get_serial_number()}")
        print(f"  - Supports depth: {camera.supports_depth()}")

        # Test RGB reading
        print("\nTesting RGB reading...")
        rgb = camera.get_rgb()
        if rgb is not None:
            print(f"  ✓ RGB image shape: {rgb.shape}, dtype: {rgb.dtype}")

            if rgb.shape[2] == 3:  # 确认是三通道图像
                pil_image = Image.fromarray(rgb)

                # 生成带时间戳的文件名，避免覆盖
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                save_path = os.path.join(save_dir, f"rgb_test_{timestamp}.jpg")

                pil_image.save(save_path, compress_level=1)
                print(f"  Success RGB image saved to: {save_path}")

            else:
                print(
                    "  Warning Image does not have 3 channels, cannot save as JPG/PNG properly."
                )
        else:
            print("  ✗ Failed to read RGB")
            return False

        # Test RGBD reading
        print("Testing RGBD reading...")
        rgbd = camera.get_rgbd()
        if rgbd is not None:
            rgb, depth = rgbd
            print(f"  ✓ RGB shape: {rgb.shape}")
            if depth is not None:
                print(
                    f"  Success Depth shape: {depth.shape}, dtype: {depth.dtype}, "
                    f"range: [{depth.min():.3f}, {depth.max():.3f}]"
                )
            else:
                print("  Warning Depth is None")
                depth = np.zeros_like(rgb[..., 0], dtype=np.float32)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

            # RGB 保存（uint8，RGB 顺序，Pillow 原生支持）
            rgb_pil = Image.fromarray(rgb)  # 直接就是 RGB
            rgb_path = os.path.join(save_dir, f"rgb_{timestamp}.png")
            rgb_pil.save(rgb_path)
            print(f"  Success RGB saved → {rgb_path}")

            if depth.dtype == np.float64 or depth.dtype == np.float32:
                # 假设深度单位是米 → 乘 1000 转为毫米，再转 uint16
                depth_mm = (depth * 1000).astype(np.uint16)
            else:
                depth_mm = depth.astype(np.uint16)

            depth_raw_path = os.path.join(
                save_dir, f"depth_raw_{timestamp}.png"
            )
            Image.fromarray(depth_mm).save(depth_raw_path)

        else:
            print("  ✗ Failed to read RGBD")

        # Test intrinsics
        print("\nTesting intrinsics...")
        color_intrinsics = camera.get_color_intrinsics()
        if color_intrinsics:
            print(
                f"  ✓ Color intrinsics: fx={color_intrinsics.get('fx'):.2f}, "
                f"fy={color_intrinsics.get('fy'):.2f}"
            )

        if camera.supports_depth():
            depth_intrinsics = camera.get_depth_intrinsics()
            if depth_intrinsics:
                print(
                    f"  ✓ Depth intrinsics: fx={depth_intrinsics.get('fx'):.2f}, "
                    f"fy={depth_intrinsics.get('fy'):.2f}"
                )

        print("\nClosing camera...")
        camera.close()
        print("  ✓ Camera closed successfully")

        return True

    except Exception as e:
        print(f"Error in single camera test: {e}")
        import traceback

        traceback.print_exc()
        return False


def example_single_camera_nonblocking(serial_number: Optional[str] = None):
    """Test: Single camera in non-blocking mode."""
    print_section("3. Testing Single Camera (Non-Blocking Mode)")

    try:
        CameraClass = Sensor("realsense")

        if serial_number is None:
            available = get_available_cameras()
            if not available:
                print("No cameras available for testing")
                return False
            serial_number = list(available.keys())[0]

        print(f"Creating camera with serial: {serial_number}")
        camera = CameraClass(
            serial_number=serial_number,
            image_width=640,
            image_height=480,
            fps=30,
            enable_depth=True,
        )

        print("Initializing camera...")
        if not camera.initialize():
            print("Failed to initialize camera")
            return False

        print("Creating non-blocking wrapper...")
        non_blocking = NonBlockingCameraWrapper(camera, frame_timeout=5000)

        print("Starting non-blocking capture...")
        if not non_blocking.start():
            print("Failed to start non-blocking capture")
            camera.close()
            return False

        print("Waiting for frames to be captured...")
        time.sleep(1.0)  # Wait for some frames to be captured

        # Test non-blocking RGB
        print("\nTesting non-blocking RGB reading...")
        for i in range(5):
            rgb = non_blocking.get_rgb()
            if rgb is not None:
                print(f"  Frame {i+1}: RGB shape {rgb.shape}")
            else:
                print(f"  Frame {i+1}: No frame available")
            time.sleep(0.1)

        # Test non-blocking RGBD
        print("\nTesting non-blocking RGBD reading...")
        for i in range(3):
            rgbd = non_blocking.get_rgbd()
            if rgbd is not None:
                rgb, depth = rgbd
                print(
                    f"  Frame {i+1}: RGB {rgb.shape}, Depth {depth.shape if depth is not None else 'None'}"
                )
            else:
                print(f"  Frame {i+1}: No frame available")
            time.sleep(0.1)

        # Test context manager
        print("\nTesting context manager...")
        with NonBlockingCameraWrapper(camera) as nb_camera:
            time.sleep(0.5)
            rgb = nb_camera.get_rgb()
            if rgb is not None:
                print(f"  ✓ Context manager works: RGB shape {rgb.shape}")
            else:
                print("  - No frame available")

        print("\nStopping non-blocking capture...")
        non_blocking.stop()
        print("Closing camera...")
        camera.close()
        print("  ✓ Test completed successfully")

        return True

    except Exception as e:
        print(f"Error in non-blocking test: {e}")
        import traceback

        traceback.print_exc()
        return False


def example_multi_camera(serial_numbers: Optional[list] = None):
    """Test: Multi-camera setup."""
    print_section("4. Testing Multi-Camera Setup")
    save_dir = "tmp"

    try:
        available = get_available_cameras()
        if not available:
            print("No cameras available for testing")
            return False

        # Use available cameras
        if serial_numbers is None:
            serial_numbers = list(available.keys())[:3]  # Use up to 3 cameras

        if len(serial_numbers) == 0:
            print("No cameras specified")
            return False

        print(f"Setting up {len(serial_numbers)} camera(s)...")

        # Create camera mapping
        cameras = {}
        roles = ["master", "left", "right"]
        for i, serial in enumerate(serial_numbers):
            if i < len(roles):
                cameras[roles[i]] = serial
                print(f"  - {roles[i]}: {serial}")

        MultiCameraClass = Sensor("multirealsense")
        multi_camera = MultiCameraClass(
            cameras=cameras, image_width=640, image_height=480, fps=30
        )

        print("\nInitializing multi-camera...")
        if not multi_camera.initialize():
            print("Failed to initialize multi-camera")
            return False

        print("Multi-camera initialized successfully")
        print(f"  - Camera roles: {multi_camera.get_camera_roles()}")

        # Test reading from all cameras
        print("\nTesting reading from all cameras...")
        all_rgb = multi_camera.get_rgb()
        if all_rgb is not None:
            if isinstance(all_rgb, np.ndarray):
                print(f"  ✓ All RGB shape: {all_rgb.shape}")
                for _rgb in all_rgb:
                    pil_image = Image.fromarray(_rgb)

                    # 生成带时间戳的文件名，避免覆盖
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
                        :-3
                    ]
                    save_path = os.path.join(
                        save_dir, f"rgb_test_{timestamp}.jpg"
                    )

                    pil_image.save(save_path, compress_level=1)
                    print(f"  Success RGB image saved to: {save_path}")

            else:
                print(f"  ✓ All RGB: {type(all_rgb)}")

        all_rgbd = multi_camera.get_rgbd()
        if all_rgbd is not None:
            rgb, depth = all_rgbd
            print(
                f"  ✓ All RGB shape: {rgb.shape if isinstance(rgb, np.ndarray) else type(rgb)}"
            )
            if depth is not None:
                print(
                    f"  ✓ All Depth shape: {depth.shape if isinstance(depth, np.ndarray) else type(depth)}"
                )

        # Test reading from specific cameras
        print("\nTesting reading from specific cameras...")
        for role in multi_camera.get_camera_roles():
            print(f"  Testing {role} camera...")
            rgb = multi_camera.get_rgb_by_role(role)
            if rgb is not None:
                print(f"    ✓ RGB shape: {rgb.shape}")

            rgbd = multi_camera.get_rgbd_by_role(role)
            if rgbd is not None:
                rgb, depth = rgbd
                print(
                    f"    ✓ RGBD: RGB {rgb.shape}, Depth {depth.shape if depth is not None else 'None'}"
                )

            camera = multi_camera.get_camera(role)
            if camera:
                print(f"    ✓ Serial: {camera.get_serial_number()}")

        # Test intrinsics
        print("\nTesting intrinsics...")
        color_intrinsics = multi_camera.get_color_intrinsics()
        if color_intrinsics:
            print(
                f"  ✓ Color intrinsics for {len(color_intrinsics)} camera(s)"
            )

        if multi_camera.supports_depth():
            depth_intrinsics = multi_camera.get_depth_intrinsics()
            if depth_intrinsics:
                print(
                    f"  ✓ Depth intrinsics for {len(depth_intrinsics)} camera(s)"
                )

        print("\nClosing multi-camera...")
        multi_camera.close()
        print("  ✓ Multi-camera closed successfully")

        return True

    except Exception as e:
        print(f"Error in multi-camera test: {e}")
        import traceback

        traceback.print_exc()
        return False


def example_rgb_only_camera(serial_number: Optional[str] = None):
    """Test: RGB-only camera (depth disabled)."""
    print_section("5. Testing RGB-Only Camera (Depth Disabled)")

    try:
        CameraClass = Sensor("realsense")

        if serial_number is None:
            available = get_available_cameras()
            if not available:
                print("No cameras available for testing")
                return False
            serial_number = list(available.keys())[0]

        print(f"Creating RGB-only camera with serial: {serial_number}")
        camera = CameraClass(
            serial_number=serial_number,
            image_width=640,
            image_height=480,
            fps=30,
            enable_depth=False,  # Disable depth
        )

        print("Initializing camera...")
        if not camera.initialize():
            print("Failed to initialize camera")
            return False

        print(f"  - Supports depth: {camera.supports_depth()}")

        # Test RGB reading
        print("\nTesting RGB reading...")
        rgb = camera.get_rgb()
        if rgb is not None:
            print(f"  ✓ RGB image shape: {rgb.shape}")
        else:
            print("  ✗ Failed to read RGB")
            return False

        # Test RGBD reading (should return None for depth)
        print("Testing RGBD reading...")
        rgbd = camera.get_rgbd()
        if rgbd is not None:
            rgb, depth = rgbd
            print(f"  ✓ RGB shape: {rgb.shape}")
            print(f"  ✓ Depth: {depth} (expected None)")
        else:
            print("  ✗ Failed to read RGBD")

        print("\nClosing camera...")
        camera.close()
        print("  ✓ Test completed successfully")

        return True

    except Exception as e:
        print(f"Error in RGB-only test: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("\n" + "=" * 80)
    print("  RealSense Camera Debug Script")
    print("=" * 80)

    # Get available cameras
    serial_numbers = example_available_cameras()

    """
        - Serial: 239222300159, Name: Intel RealSense D455
        - Serial: 241122305539, Name: Intel RealSense D455
    """

    if not serial_numbers:
        print("\nNo cameras available. Exiting.")
        return

    # Run tests
    results = {}

    # Test 1: Single camera blocking
    if serial_numbers:
        results["single_blocking"] = example_single_camera_blocking(
            serial_numbers[0]
        )

    # Test 2: Single camera non-blocking
    if serial_numbers:
        results["single_nonblocking"] = example_single_camera_nonblocking(
            serial_numbers[0]
        )

    # Test 3: Multi-camera (if multiple cameras available)
    if len(serial_numbers) > 1:
        results["multi_camera"] = example_multi_camera(serial_numbers)
    else:
        print_section("4. Testing Multi-Camera Setup")
        print("Skipped: Need at least 2 cameras for multi-camera test")
        results["multi_camera"] = None

    # Test 4: RGB-only camera
    if serial_numbers:
        results["rgb_only"] = example_rgb_only_camera(serial_numbers[0])

    # Print summary
    print_section("Test Summary")
    for test_name, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"  {test_name:20s}: {status}")

    print("\n" + "=" * 80)
    print("  Debug script completed")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
