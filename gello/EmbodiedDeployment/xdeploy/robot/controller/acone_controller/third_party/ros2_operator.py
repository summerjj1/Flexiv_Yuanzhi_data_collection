import collections
import os
import threading
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R  # eef:ZXY
from std_msgs.msg import Int32MultiArray
from tf2_msgs.msg import TFMessage


class RosOperator(Node):
    def __init__(self, args, config, in_collect=False):
        super().__init__("robot_operator")

        from arm_control.msg._pos_cmd import PosCmd
        from arx5_arm_msg.msg._robot_status import RobotStatus

        self.args = args
        self.config = config

        self.in_collect = in_collect

        self.base_enable = False
        self.robot_base_pose_init = [
            0,
            0,
            0,
        ]  # rlative, the head_pitch and height and head yaw is the adsolutly
        self.robot_base_target = np.zeros((6,))
        self.base_velocity_target = np.zeros((4,))
        self.base_control_thread = None

        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()

        self.bridge = CvBridge()

        self.controller_left_deque = deque()
        self.controller_right_deque = deque()
        self.feedback_left_arm_deque = deque()
        self.feedback_right_arm_deque = deque()

        self.base_pose_deque = deque()
        self.robot_base_origin = deque()
        self.robot_base_deque = deque()
        self.base_velocity_deque = deque()

        self.follow_arm_publish_lock = threading.Lock()
        self.follow_arm_publish_lock.acquire()

        self.last_joy = [0, 0, 0, 0]
        self.triggered_joys = {}
        self.joy_lock = threading.Lock()

        self.pos_cmd = PosCmd
        self.robot_status = RobotStatus

        # 机械臂订阅
        arm_topics = {
            "feedback_left": ("feedback_left_topic", self.robot_status),
            "feedback_right": ("feedback_right_topic", self.robot_status),
        }

        if self.in_collect:
            arm_topics.update(
                {
                    "controller_left": (
                        "controller_left_topic",
                        self.robot_status,
                    ),
                    "controller_right": (
                        "controller_right_topic",
                        self.robot_status,
                    ),
                }
            )

        for key, (topic_key, msg_type) in arm_topics.items():
            try:
                self.create_subscription(
                    msg_type,
                    self.config["arm_config"][topic_key],
                    getattr(self, f"{key}_callback"),
                    2,
                )
            except KeyError as e:
                self.get_logger().error(f"Topic config missing: {e}")
            except AttributeError as e:
                self.get_logger().error(
                    f"Callback not found for key: {key} -> {e}"
                )

        # 按键订阅
        self.create_subscription(
            Int32MultiArray,
            self.config["joy_config"]["joy_topic"],
            self.joy_callback,
            2,
        )

        # 底盘订阅
        if self.args.use_base:
            self.create_subscription(
                self.pos_cmd,
                self.config["robot_base_config"]["robot_base_topic"],
                self.robot_base_callback,
                2,
            )

            if self.args.record == "Distance":
                self.create_subscription(
                    TFMessage, "/tf", self.base_pose_callback, 2
                )
            if self.args.record == "Speed":
                self.create_subscription(
                    self.pos_cmd,
                    self.config["robot_base_config"]["robot_base_topic"],
                    self.base_velocity_callback,
                    2,
                )
        # 推理模式相关发布
        if not self.in_collect:
            self.controller_arm_left_publisher = self.create_publisher(
                self.robot_status,
                self.config["arm_config"]["controller_left_topic"],
                10,
            )
            self.controller_arm_right_publisher = self.create_publisher(
                self.robot_status,
                self.config["arm_config"]["controller_right_topic"],
                10,
            )
            self.base_actuator_publisher = self.create_publisher(
                self.pos_cmd,
                self.config["robot_base_config"]["robot_base_cmd_topic"],
                10,
            )

    # 推理
    def follow_arm_publish(self, left, right):
        if len(left) == 7:
            joint_state_msg = self.robot_status()
        else:
            print("\033[31mERROR action\033[0m")

            return
        if not isinstance(left, np.ndarray):
            left = np.array(left)
        if not isinstance(right, np.ndarray):
            right = np.array(right)
        joint_state_msg.joint_pos = left.astype(np.float64)
        self.controller_arm_left_publisher.publish(
            joint_state_msg
        )  # /joint_control
        if len(right) != 0:
            joint_state_msg.joint_pos = right.astype(np.float64)
            self.controller_arm_right_publisher.publish(
                joint_state_msg
            )  # /joint_control2

    def init_robot_base_pose(self):
        if len(self.robot_base_origin) == 0:
            print(r"there is no base_pose_deque")

            return None
        base_pose = self.robot_base_origin.pop()
        tf_info = base_pose.transforms[0].transform
        base_quaternion = [
            tf_info.rotation.x,
            tf_info.rotation.y,
            tf_info.rotation.z,
            tf_info.rotation.w,
        ]
        r = R.from_quat(base_quaternion)
        _, _, base_pose_yaw = r.as_euler("xyz", degrees=False)
        base_pose = [
            tf_info.translation.x,
            -tf_info.translation.y,
            base_pose_yaw,
        ]
        self.robot_base_pose_init = base_pose

        self.robot_base_target = np.zeros((6,))

        return True

    def set_robot_base_target(self, target_base):
        self.robot_base_target[0] = target_base[0]  # x
        self.robot_base_target[1] = target_base[1]  # y
        self.robot_base_target[2] = target_base[2]  # Wz
        self.robot_base_target[3] = target_base[3]  # height
        self.robot_base_target[4] = target_base[4]  # head_pitch
        self.robot_base_target[5] = target_base[5]  # head_yaw

        self.base_velocity_target[0] = target_base[6]  # motor1
        self.base_velocity_target[1] = target_base[7]  # motor2
        self.base_velocity_target[2] = target_base[8]  # motor3
        self.base_velocity_target[3] = target_base[9]  # motor4

    def start_base_control_thread(self):
        if self.args.use_base:
            self.init_robot_base_pose()
            self.base_enable = True
            self.base_control_thread = threading.Thread(
                target=self.robot_base_control_thread, args=()
            )  # 执行指令单独的线程,，可以边说话边执行，多线程操作
            self.base_control_thread.start()

            return

    def visualize_pid_base(self, states, target, plot_path=None):
        STATE_NAMES = ["DX", "DY", "Yaw"]
        label1, label2 = "states", "target"
        states = np.array(states)
        target = np.array(target)

        num_ts, num_dim = states.shape
        fig, axs = plt.subplots(num_dim, 1, figsize=(8, 2 * num_dim))

        all_names = [f"{name}_left" for name in STATE_NAMES] + [
            f"{name}_right" for name in STATE_NAMES
        ]

        for dim_idx, ax in enumerate(axs):
            ax.plot(states[:, dim_idx], label=label1, color="orangered")
            ax.plot(target[:, dim_idx], label=label2)
            ax.set_title(f"Joint {dim_idx}: {all_names[dim_idx]}")
            ax.legend()

        plt.tight_layout()
        if plot_path:
            plt.savefig(plot_path)
            print(f"Saved pid control plot to: {plot_path}")
        else:
            plt.show()

        plt.close()

    def robot_base_shutdown(self):
        rate = self.create_rate(self.args.frame_rate)

        shutdown_control = self.pos_cmd()
        shutdown_control.height = self.robot_base_target[3]

        for mode in [1, 2]:
            shutdown_control.mode1 = mode
            self.base_actuator_publisher.publish(shutdown_control)

            rate.sleep()

        self.base_enable = False

        return

    def follow_arm_publish_continuous(self, left_target, right_target):
        arm_steps_length = [0.05, 0.05, 0.03, 0.05, 0.05, 0.05, 0.2]
        left_arm = None
        right_arm = None

        rate = self.create_rate(self.args.frame_rate)
        while rclpy.ok():
            if len(self.feedback_left_arm_deque) != 0:
                left_arm = list(self.feedback_left_arm_deque[-1].joint_pos)

            if len(self.feedback_right_arm_deque) != 0:
                right_arm = list(self.feedback_right_arm_deque[-1].joint_pos)

            if left_arm is not None and right_arm is not None:
                break

        # 计算方向标志位
        left_symbol = [
            1 if left_target[i] - left_arm[i] > 0 else -1
            for i in range(len(left_target))
        ]
        right_symbol = [
            1 if right_target[i] - right_arm[i] > 0 else -1
            for i in range(len(right_target))
        ]

        step = 0
        while rclpy.ok():
            left_done = 0
            right_done = 0

            if self.follow_arm_publish_lock.acquire(False):
                return

            left_done = self._update_arm_position(
                left_target, left_arm, left_symbol, arm_steps_length
            )
            right_done = self._update_arm_position(
                right_target, right_arm, right_symbol, arm_steps_length
            )

            if (
                left_done > len(left_target) - 1
                and right_done > len(right_target) - 1
            ):
                print("left_done and right_done")

                break

            # JointControl topic
            if len(left_arm) == 7:
                joint_state_msg = self.robot_status()
            else:
                print("\033[31mInvalid joint length\033[0m")

                return

            joint_state_msg.joint_pos = np.asarray(left_arm, dtype=np.float64)
            self.controller_arm_left_publisher.publish(joint_state_msg)
            rate.sleep()

            joint_state_msg.joint_pos = np.asarray(right_arm, dtype=np.float64)
            self.controller_arm_right_publisher.publish(joint_state_msg)

            step += 1
            print("arm_publish_continuous:", step)
            rate.sleep()

    def _extract_eef_data(self, eef):
        return [eef.x, eef.y, eef.z, eef.roll, eef.pitch, eef.yaw]

    def get_observation(self, ts=-1):  # get the robot observation

        arm_data = {
            "left_arm": self.robot_status(),
            "right_arm": self.robot_status(),
        }

        # 获取机械臂状态
        for arm_name in ["left_arm", "right_arm"]:
            deque_map = {
                "left_arm": self.feedback_left_arm_deque,
                "right_arm": self.feedback_right_arm_deque,
            }

            if len(deque_map[arm_name]) == 0:
                print(f"there is no {arm_name}_deque")

                return None

            arm_data[arm_name] = deque_map[arm_name].pop()

        obs_dict = collections.OrderedDict()  # 有序的字典

        # 保存机械臂状态
        left_eef = np.concatenate(
            [
                arm_data["left_arm"].end_pos,
                [arm_data["left_arm"].joint_pos[-1]],
            ]
        )

        right_eef = np.concatenate(
            [
                arm_data["right_arm"].end_pos,
                [arm_data["right_arm"].joint_pos[-1]],
            ]
        )

        obs_dict["eef"] = np.concatenate((left_eef, right_eef), axis=0)
        obs_dict["qpos"] = np.concatenate(
            (
                np.array(arm_data["left_arm"].joint_pos),
                np.array(arm_data["right_arm"].joint_pos),
            ),
            axis=0,
        )
        obs_dict["qvel"] = np.concatenate(
            (
                np.array(arm_data["left_arm"].joint_vel),
                np.array(arm_data["right_arm"].joint_vel),
            ),
            axis=0,
        )
        obs_dict["effort"] = np.concatenate(
            (
                np.array(arm_data["left_arm"].joint_cur),
                np.array(arm_data["right_arm"].joint_cur),
            ),
            axis=0,
        )

        # 保存底盘状态
        if self.args.use_base and ts != 0:
            if len(self.robot_base_deque) == 0:
                print(
                    r"there is no robot_base_deque, maby there is no VR message"
                )

                return None

            if self.args.record == "Distance":
                if len(self.base_pose_deque) == 0:
                    print(r"there is no base_pose_deque")

                    return None
            if self.args.record == "Speed":
                if len(self.base_velocity_deque) == 0:
                    print(r"there is no base_velocity_deque")

                    return None

            robot_base = self.robot_base_deque.pop()

            if self.args.record == "Distance":
                base_pose = self.base_pose_deque.pop()
                obs_dict["robot_base"] = [
                    base_pose[0],
                    base_pose[1],
                    base_pose[2],
                    robot_base.height,
                    robot_base.head_pit,
                    robot_base.head_yaw,
                ]

                obs_dict["base_velocity"] = np.zeros((4,))
            if self.args.record == "Speed":
                obs_dict["robot_base"] = [
                    0,
                    0,
                    0,
                    robot_base.height,
                    robot_base.head_pit,
                    robot_base.head_yaw,
                ]

                base_velocity = self.base_velocity_deque.pop()
                obs_dict["base_velocity"] = [
                    base_velocity[0],
                    base_velocity[1],
                    base_velocity[2],
                    base_velocity[3],
                ]
        else:
            obs_dict["robot_base"] = np.zeros((6,))
            obs_dict["base_velocity"] = np.zeros((4,))

        return obs_dict

    def get_action(self):
        joints_dim = 7

        action_dict = collections.OrderedDict()

        deque_map = {
            "control_left_arm_deque": self.controller_left_deque,
            "control_right_arm_deque": self.controller_right_deque,
        }

        for name, deque in deque_map.items():
            if len(deque) == 0:
                print(f"there is no {name}")

                return None

        # 获取主臂状态
        left_frame = deque_map["control_left_arm_deque"].pop()
        right_frame = deque_map["control_right_arm_deque"].pop()

        control_left_arm = left_frame.end_pos
        control_right_arm = right_frame.end_pos
        control_left_arm_gripper = left_frame.joint_pos[-1]
        control_right_arm_gripper = right_frame.joint_pos[-1]

        # 主臂保存状态
        control_left_arm_eef = np.concatenate(
            [control_left_arm, [control_left_arm_gripper]]
        )
        control_right_arm_eef = np.concatenate(
            [control_right_arm, [control_right_arm_gripper]]
        )

        # 构建动作字典
        action_dict["action"] = np.zeros((joints_dim * 2,))
        action_dict["action_qvel"] = np.zeros((joints_dim * 2,))
        action_dict["action_eef"] = np.concatenate(
            (control_left_arm_eef, control_right_arm_eef), axis=0
        )
        action_dict["action_base"] = np.zeros(
            (13,)
        )  # waiting for the obersevation

        return action_dict

    def img_head_callback(self, msg):
        if len(self.img_head_deque) >= 2000:
            self.img_head_deque.popleft()
        self.img_head_deque.append(msg)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_head_depth_callback(self, msg):
        if len(self.img_head_depth_deque) >= 2000:
            self.img_head_depth_deque.popleft()
        self.img_head_depth_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def controller_left_callback(self, msg):
        if len(self.controller_left_deque) >= 2000:
            self.controller_left_deque.popleft()
        self.controller_left_deque.append(msg)
        self.feedback_left_arm_deque.append(msg)

    def controller_right_callback(self, msg):
        if len(self.controller_right_deque) >= 2000:
            self.controller_right_deque.popleft()
        self.controller_right_deque.append(msg)
        self.feedback_right_arm_deque.append(msg)

    def feedback_left_callback(self, msg):
        if len(self.feedback_left_arm_deque) >= 2000:
            self.feedback_left_arm_deque.popleft()
        self.feedback_left_arm_deque.append(msg)

    def feedback_right_callback(self, msg):
        if len(self.feedback_right_arm_deque) >= 2000:
            self.feedback_right_arm_deque.popleft()
        self.feedback_right_arm_deque.append(msg)

    # robot robot_base
    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def base_pose_callback(self, msg):
        if len(self.base_pose_deque) >= 2:
            self.base_pose_deque.popleft()

        if len(self.robot_base_origin) >= 2:
            self.robot_base_origin.popleft()
        self.robot_base_origin.append(msg)

        tf_info = msg.transforms[0].transform
        base_quaternion = [
            tf_info.rotation.x,
            tf_info.rotation.y,
            tf_info.rotation.z,
            tf_info.rotation.w,
        ]
        r = R.from_quat(base_quaternion)
        _, _, base_pose_yaw = r.as_euler("xyz", degrees=False)
        base_pose = [
            tf_info.translation.x,
            -tf_info.translation.y,
            base_pose_yaw,
        ]

        base_pose[0] = base_pose[0] - self.robot_base_pose_init[0]  # 如果这个值是负的
        base_pose[1] = base_pose[1] - self.robot_base_pose_init[1]
        base_pose[2] = base_pose[2] - self.robot_base_pose_init[2]

        self.base_pose_deque.append(base_pose)

    def base_velocity_callback(self, msg):
        if len(self.base_velocity_deque) >= 2:
            self.base_velocity_deque.popleft()

        velocity = msg.temp_float_data[1:5]

        self.base_velocity_deque.append(velocity)

    def joy_callback(self, msg):
        joy = list(msg.data)

        with self.joy_lock:
            for i in range(4):
                if self.last_joy[i] == 0 and joy[i] == 1:
                    self.triggered_joys[i] = joy.copy()

            self.last_joy = joy

    def _update_arm_position(self, target, arm, symbol, steps_length):
        diff = [abs(target[i] - arm[i]) for i in range(len(target))]
        done = 0
        for i in range(len(target)):
            if diff[i] < steps_length[i]:
                arm[i] = target[i]
                done += 1
            else:
                arm[i] += symbol[i] * steps_length[i]

        return done
