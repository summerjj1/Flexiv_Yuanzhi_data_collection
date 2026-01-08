
#!/usr/bin/env python
import sys
import time
import argparse
import spdlog
import flexivrdk
import numpy as np
from tqdm import tqdm

import pyarrow.parquet as pq
import os

# —— 配置区域 —— #
parquet_path = "/media/forceyqj/My PSSD/processed_data/middle_gear_multi_task/data/chunk-000/episode_000406.parquet"

# 只读需要的列（这里只用 action 即可；observation.state 可按需再加）
table = pq.read_table(parquet_path, columns=["action_and_force_params"])
actions_ca = table.column("action_and_force_params")

# import pdb
# pdb.set_trace()
# table = pq.read_table(parquet_path, columns=["action"])
# actions_ca = table.column("action")

table_k = pq.read_table(parquet_path, columns=["action.stiffness_list"])
actions_ca_k = table_k.column("action.stiffness_list")

table_f = pq.read_table(parquet_path, columns=["action.wrench_T_des"])
actions_ca_f = table_f.column("action.wrench_T_des")

records_f_list = actions_ca_f.to_pylist()
records_f = []
for a in records_f_list:
    if isinstance(a, list) and len(a) == 1 and isinstance(a[0], (list, tuple, np.ndarray)):
        a = a[0]
    records_f.append(a)
records_f = np.asarray(records_f, dtype=np.float32)

records_k_list = actions_ca_k.to_pylist()
records_k = []

for a in records_k_list:
    if isinstance(a, list) and len(a) == 1 and isinstance(a[0], (list, tuple, np.ndarray)):
        a = a[0]
    records_k.append(a)
records_k = np.asarray(records_k, dtype=np.float32)

records_list = actions_ca.to_pylist()
records = []



for a in records_list:
    if isinstance(a, list) and len(a) == 1 and isinstance(a[0], (list, tuple, np.ndarray)):
        a = a[0]
    records.append(a)

# 转成 (N, 8) 数组，前7维是位姿 [x,y,z,qx,qy,qz,qw]，第8维是 gripper
records = np.asarray(records, dtype=np.float32)
def main(args):
    """Main function to control the Flexiv Rizon 4s robot and execute a trajectory."""
    logger = spdlog.ConsoleLogger("Robot Control Program")
    mode = flexivrdk.Mode

    robot = None
    gripper = None
    try:
        # Initialize robot
        logger.info("Connecting to robot...")
        robot = flexivrdk.Robot(args.robot_serial)
        
        # Check and clear any faults
        if robot.fault():
            logger.warn("Robot fault detected, attempting to clear...")
            robot.ClearFault()
            time.sleep(0.1)
            if robot.fault():
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
        gripper.Move(0.1, 0.1, 50)
        while robot.busy():
            time.sleep(0.1)

        # # Execute home plan
        # robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
        # robot.ExecutePlan("PLAN-Home")
        # logger.info("Executing home plan...")
        # while robot.busy():
        #     time.sleep(0.1)
        # logger.info("Home plan completed")

        robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
        robot.ExecutePrimitive("ZeroFTSensor", dict())
        logger.warn("Zeroing force/torque sensors, make sure nothing is in contact with the robot")
        while not robot.primitive_states()["terminated"]:
            time.sleep(0.1)
        logger.info("Sensor zeroing complete")

        robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)

        # —— 从 Parquet 的 action 构造轨迹 —— #
        acts = records  
        print(acts[0])
        traj_pose = acts[:, :7]    
        traj_gripper = acts[:, 7] 
        force_list = acts[:, 8:14]    

        # sitffiness_list = records_k[:, :] * 5.3

        # force_control_list = records_f[:, :]
        logger.info("Executing trajectory (from Parquet 'action')...")
        index = 0
        force_control = True
        last_force_control_mode_flag = False
        for pose, gripper_width, force in tqdm(zip(traj_pose, traj_gripper, force_list), total=len(traj_gripper)):
            # robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)

            # index +=1
            # if index < 350:
            #     continue
            # force_control_mode_flag = True if abs(force[2]) > 5 else False

            force_control_mode_flag = False
            # pose[2] = pose[2] + 0.15

            if not force_control_mode_flag:
                print('Postion------')
                # robot.SendCartesianMotionForce(new_target, [0]*6)
                robot.SetForceControlAxis([False, False, False, False, False, False])
                robot.SendCartesianMotionForce(
                        [float(pose[0]), float(pose[1]), float(pose[2]),
                        float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])],
                        [0.0] * 6,
                    )
            else:
                print('force------')
                SEARCH_VELOCITY = 0.2

                robot.SetForceControlAxis([True, True, True, False, False, False])
                robot.SendCartesianMotionForce(pose,force, SEARCH_VELOCITY, max_linear_acc = 0.1, max_angular_acc = 0.5)

            gripper.Move(float(gripper_width), 0.1, 50)
            

            time.sleep(1 / 20)  # 如无时间戳，保持固定节拍

            #ADD
            # Switch to primitive execution mode
            # robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
            # # Send command to robot
            # robot.ExecutePrimitive("Hold", dict())

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            if gripper is not None:
                # Apply zero force control if gripper force is detected
                try:
                    if abs(gripper.states().force) > sys.float_info.epsilon:
                        logger.info("Applying zero force control to gripper...")
                        gripper.Grasp(0)
                        time.sleep(0.1)
                except Exception:
                    pass

                logger.info("Stopping gripper...")
                gripper.Stop()
        except Exception:
            pass

        try:
            if robot is not None:
                logger.info("Returning to home position...")
                robot.SwitchMode(mode.NRT_PLAN_EXECUTION)
                robot.ExecutePlan("PLAN-Home")
                while robot.busy():
                    time.sleep(0.1)
                logger.info("Program completed")
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control script for Flexiv Rizon 4s robot with gripper.")
    parser.add_argument('--robot_serial', type=str, default="Rizon 4s-063036", 
                        help="Serial number of the robot (default: Rizon 4s-063036)")
    parser.add_argument('--gripper_name', type=str, default="GripperDahuanModbus", 
                        help="Name of the gripper (default: GripperDahuanModbus)")
    # 这里保留了 h5_file 参数但未使用；如果你只打算用 Parquet，可以删掉它
    parser.add_argument('--h5_file', type=str, default="/home/liyang/data/flexiv/teleop_recordings/2025-09-26/trajectory_20250926_205354.h5",
                        help="Path to the HDF5 file containing trajectory data (unused when reading Parquet)")
    args = parser.parse_args()

    main(args)
