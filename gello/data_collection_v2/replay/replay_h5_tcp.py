#!/usr/bin/env python
import sys
import time
import argparse
import spdlog
import flexivrdk
import numpy as np
import h5py
from tqdm import tqdm
import quaternion


def main(args):
    """Main function to control the Flexiv Rizon 4s robot and execute a trajectory."""
    # Initialize logger
    logger = spdlog.ConsoleLogger("Robot Control Program")
    mode = flexivrdk.Mode

    try:
        # Initialize robot
        logger.info("Connecting to robot...")
        robot = flexivrdk.Robot(args.robot_serial)
        
        # Check and clear any faults
        if robot.fault():
            logger.warn("Robot fault detected, attempting to clear...")
            robot.ClearFault()
            time.sleep(0.1)
            if not robot.ClearFault():
                logger.error("Unable to clear robot fault. Exiting...")
                return
            logger.info("Robot fault cleared successfully")

        # Enable robot
        logger.info("Enabling robot...")
        robot.Enable()
        while not robot.operational():
            time.sleep(0.1)
        logger.info("Robot is now operational")

        # Initialize and enable gripper
        logger.info(f"Enabling gripper '{args.gripper_name}'...")
        gripper = flexivrdk.Gripper(robot)
        gripper.Enable(args.gripper_name)

        # Initialize tool
        tool = flexivrdk.Tool(robot)

        # Open gripper
        logger.info("Opening gripper...")
        gripper.Move(0.0, 0.1, 50)
        while robot.busy():
            time.sleep(0.1)

        # Execute home plan
        # robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
        robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)
        # robot.ExecutePlan("PLAN-Home")
        # logger.info("Executing home plan...")
        # while robot.busy():
        #     time.sleep(0.1)
        # logger.info("Home plan completed")

        # robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
        # robot.ExecutePrimitive("ZeroFTSensor", dict())
        # logger.warn(
        #     "Zeroing force/torque sensors, make sure nothing is in contact with the robot"
        # )
        # while not robot.primitive_states()["terminated"]:
        #     time.sleep(0.1)
        # logger.info("Sensor zeroing complete")

        home_target= [0.69537903, -0.15347707, 0.42430178, 0.01960641353, 0.057527819, 0.9980455, -0.01453]
        cmd_end_effector_pos = home_target[:3]
        cmd_quat_numpy = quaternion.quaternion(home_target[3], home_target[4], home_target[5], home_target[6])
        print("Homing to position:", cmd_end_effector_pos)
        print("Homing to quaternion:", cmd_quat_numpy)
        robot.SendCartesianMotionForce([*cmd_end_effector_pos, cmd_quat_numpy.w, cmd_quat_numpy.x, cmd_quat_numpy.y, cmd_quat_numpy.z])
        logger.warn(
            "Zeroing force/torque sensors, make sure nothing is in contact with the robot"
        )

        # while not robot.primitive_states()["terminated"]:
        #     time.sleep(0.1)
        # logger.info("Sensor zeroing complete")


        # robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)

        # Load trajectory from HDF5 file
        logger.info(f"Loading trajectory from {args.h5_file}...")
        with h5py.File(args.h5_file, 'r') as f:
            traj_pose = f['action'][:,:7]  # Assuming first 7 values are pose
            traj_gripper = f['gripper_width'][:]
            # traj_gripper = f['action'][:,-1]
        # import pdb; pdb.set_trace()
        # Execute trajectory
        logger.info("Executing trajectory...")
        for pose, gripper_width in tqdm(zip(traj_pose, traj_gripper)):
            robot.SendCartesianMotionForce(
                [*pose[:3], pose[3], pose[4], pose[5], pose[6]],
                [0.0] * 6,
            )
            if gripper_width > 0.5:
                gripper.Move(0.14, 0.1, 50)
            else:
                gripper.Move(0, 0.1, 50)
            # print(pose, gripper_width)
            # import pdb;pdb.set_trace()
            time.sleep(1 / 30)  # Adjust based on trajectory timing

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Apply zero force control if gripper force is detected
        if abs(gripper.states().force) > sys.float_info.epsilon:
            logger.info("Applying zero force control to gripper...")
            gripper.Grasp(0)
            # time.sleep(0.1)

        # Stop gripper
        logger.info("Stopping gripper...")
        gripper.Stop()

        # Return to home position
        # logger.info("Returning to home position...")
        # robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
        # robot.ExecutePlan("PLAN-Home")




        robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)
        home_target= [0.69537903, -0.15347707, 0.42430178, 0.01960641353, 0.057527819, 0.9980455, -0.01453]
        cmd_end_effector_pos = home_target[:3]
        cmd_quat_numpy = quaternion.quaternion(home_target[3], home_target[4], home_target[5], home_target[6])
        print("Homing to position:", cmd_end_effector_pos)
        print("Homing to quaternion:", cmd_quat_numpy)
        robot.SendCartesianMotionForce([*cmd_end_effector_pos, cmd_quat_numpy.w, cmd_quat_numpy.x, cmd_quat_numpy.y, cmd_quat_numpy.z])

        while robot.busy():
            time.sleep(0.1)
        logger.info("Program completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control script for Flexiv Rizon 4s robot with gripper.")
    parser.add_argument('--robot_serial', type=str, default="Rizon 4 -00015", 
                        help="Serial number of the robot (default: Rizon 4s-063036)")
    parser.add_argument('--gripper_name', type=str, default="GripperDahuanModbus", 
                        help="Name of the gripper (default: GripperDahuanModbus)")
    parser.add_argument('--h5_file', type=str, default="/media/ubuntu/ZhqSSD/YZdata/insert_hole_twocam_24/2025-12-30/trajectory_20251230_203441.h5",
                        help="Path to the HDF5 file containing trajectory data")
    args = parser.parse_args()

    main(args)