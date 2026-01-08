
import cv2
from xdeploy.robot.sensor import Sensor
from xdeploy.robot.sensor.camera.realsense import get_available_cameras




if __name__=="__main__":

        # - right: 313522302661
        # - hand: 247122072727
        # - left: 336522303555
        available = get_available_cameras()
        if not available:
            print("No cameras available for testing")

        # Use available cameras
        serial_numbers = list(available.keys())[:3]  # Use up to 3 cameras

        if len(serial_numbers) == 0:
            print("No cameras specified")

        print(f"Setting up {len(serial_numbers)} camera(s)...")

        # Create camera mapping
        cameras = {}
        roles = ["left", "right", "hand"]
        for i, serial in enumerate(serial_numbers):
            if i < len(roles):
                cameras[roles[i]] = serial
                print(f"  - {roles[i]}: {serial}")

        MultiCameraClass = Sensor("multirealsense")
        multi_camera = MultiCameraClass(
            cameras=cameras, image_width=640, image_height=480, fps=30
        )
        multi_camera.initialize()

        # get images
        all_rgb = multi_camera.get_rgb()

        # visualize 
        print("all_rgb", all_rgb.shape)
        cv2.imshow("left", all_rgb[0]) # hand 
        cv2.imshow("right", all_rgb[1]) # left right
        cv2.imshow("hand", all_rgb[2]) # right hand 
        cv2.waitKey(0)


