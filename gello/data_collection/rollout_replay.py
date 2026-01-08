#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
flexiv_ros_teleop_slam_vive_clamp.py

使用 ROS 的 slam / vive / clamp topic 遥操作 Flexiv 机械臂 + 夹爪。
- 位姿来源: slam 或 vive（二选一，通过 --source 指定）
- 夹爪来源: clamp（XVisio 夹爪）

依赖:
    - flexivrdk
    - api.flexiv_robot.flexiv_robot
    - pose_merge.transform_vive_to_gripper, transform_slam_to_gripper
    - ROS: rospy, geometry_msgs/PoseStamped, xv_sdk/PoseStampedConfidence, roslib

python flexiv_ros_teleop_slam_vive_clamp.py \
    --xv-serial 250801DR48FP25002269 \
    --vive-serial LHR-12345678 \
    --source slam \
    --clamp-max 88
"""

import argparse
import threading
import time

import numpy as np
import flexivrdk
import rospy
from geometry_msgs.msg import PoseStamped
from xv_sdk.msg import PoseStampedConfidence
from roslib.message import get_message_class

from pose_merge import transform_vive_to_gripper, transform_slam_to_gripper
from api.flexiv_robot import flexiv_robot


class FlexivRosTeleopController:
    """
    使用 ROS topic (slam / vive / clamp) 遥操作 Flexiv 机械臂。

    - slam_topic : /xv_sdk/<xv_serial>/slam/pose           (xv_sdk/PoseStampedConfidence)
    - vive_topic : /vive/<vive_serial>/pose                (geometry_msgs/PoseStamped)
    - clamp_topic: /xv_sdk/<xv_serial>/clamp/Data          (任意消息，动态解析出数值)

    pose_source ∈ {"slam", "vive"} 决定用哪个位姿源。
    """

    def __init__(
        self,
        robot_wrapper,
        xv_serial: str,
        vive_serial: str,
        pose_source: str = "vive",
        clamp_max_open: float = 88.0,
        max_linear_vel: float = 0.25,
        max_angular_vel: float = 1.5,
    ):
        assert pose_source in ("slam", "vive")
        self.robot_wrapper = robot_wrapper
        self.robot = robot_wrapper.robot
        self.pose_source = pose_source
        self.xv_serial = xv_serial
        self.vive_serial = vive_serial
        self.clamp_max_open = float(clamp_max_open)
        self.max_linear_vel = float(max_linear_vel)
        self.max_angular_vel = float(max_angular_vel)

        # Topic 名称
        self.slam_topic = f"/xv_sdk/{self.xv_serial}/slam/pose"
        vive_serial_safe = self.vive_serial.replace("-", "_")
        self.vive_topic = f"/vive/{vive_serial_safe}/pose"
        self.clamp_topic = f"/xv_sdk/{self.xv_serial}/clamp/Data"

        # 机械臂命令节流
        self.robot_lock = threading.Lock()
        self.last_arm_cmd_time = 0.0
        self.arm_cmd_interval = 0.01  # 最多 100 Hz 发送笛卡尔命令

        # 夹爪控制节流
        self.clamp_msg_class = None
        self.last_gripper_cmd = None  # 上一次发送的开合比例 [0,1]
        self.last_gripper_cmd_time = 0.0
        self.gripper_deadband = 0.02      # 小于 2% 变化就不发
        self.gripper_cmd_interval = 0.05  # 最多 20 Hz 发一次

        # 记录当前夹爪比例，方便以后接录制器用
        self.current_gripper_ratio = 1.0  # 默认全开/全闭视你封装而定

        rospy.loginfo("FlexivRosTeleopController 初始化")
        rospy.loginfo("  slam topic  : %s", self.slam_topic)
        rospy.loginfo("  vive topic  : %s", self.vive_topic)
        rospy.loginfo("  clamp topic : %s", self.clamp_topic)
        rospy.loginfo("  pose source : %s", self.pose_source)

        self._check_topics()
        self._subscribe_topics()

    # ------------------------------------------------------------------ #
    # ROS Topic 检查 & 订阅
    # ------------------------------------------------------------------ #
    def _check_topics(self):
        """简单检查一下关键话题是否已经被发布（仅提示，不阻止运行）"""
        try:
            topics = rospy.get_published_topics()
            topic_names = [t[0] for t in topics]
        except Exception as e:
            rospy.logwarn("获取 topic 列表失败: %s", e)
            return

        missing = []
        if self.pose_source == "slam":
            if self.slam_topic not in topic_names:
                missing.append(self.slam_topic)
        elif self.pose_source == "vive":
            if self.vive_topic not in topic_names:
                missing.append(self.vive_topic)
        if self.clamp_topic not in topic_names:
            rospy.logwarn("未发现 clamp 话题: %s（仍然继续运行）", self.clamp_topic)

        if missing:
            rospy.logwarn("以下控制用话题当前没有发布者（可以稍后再启动）：")
            for t in missing:
                rospy.logwarn("  - %s", t)

    def _subscribe_topics(self):
        # slam 位姿
        self.slam_sub = rospy.Subscriber(
            self.slam_topic,
            PoseStampedConfidence,
            self._slam_callback,
            queue_size=200,
            buff_size=2**20,
            tcp_nodelay=True,
        )
        # vive 位姿
        self.vive_sub = rospy.Subscriber(
            self.vive_topic,
            PoseStamped,
            self._vive_callback,
            queue_size=100,
            buff_size=2**20,
            tcp_nodelay=True,
        )
        # clamp 夹爪（AnyMsg 动态解析）
        self.clamp_sub = rospy.Subscriber(
            self.clamp_topic,
            rospy.AnyMsg,
            self._clamp_callback,
            queue_size=100,
            buff_size=2**20,
            tcp_nodelay=True,
        )

    # ------------------------------------------------------------------ #
    # 回调：位姿来源
    # ------------------------------------------------------------------ #
    def _slam_callback(self, msg: PoseStampedConfidence):
        """SLAM 位姿 -> gripper pose -> Flexiv base pose"""
        if self.pose_source != "slam":
            return
        try:
            pose_msg = msg.poseMsg.pose
            ts = msg.poseMsg.header.stamp.to_sec()

            x = pose_msg.position.x
            y = pose_msg.position.y
            z = pose_msg.position.z
            qx = pose_msg.orientation.x
            qy = pose_msg.orientation.y
            qz = pose_msg.orientation.z
            qw = pose_msg.orientation.w

            qpos_xv = [x, y, z, qx, qy, qz, qw]
            # 假设 transform_slam_to_gripper 已经把坐标系对齐到 Flexiv base
            gx, gy, gz, gqx, gqy, gqz, gqw = transform_slam_to_gripper(qpos_xv)

            # Flexiv RDK 要求四元数顺序 [w, x, y, z]
            cmd_pose = [
                float(gx),
                float(gy),
                float(gz),
                float(gqw),
                float(gqx),
                float(gqy),
                float(gqz),
            ]
            self._send_cartesian_cmd(cmd_pose, ts, src="slam")
        except Exception as e:
            rospy.logerr("处理 SLAM 位姿时出错: %s", e)

    def _vive_callback(self, msg: PoseStamped):
        """Vive 位姿 -> gripper pose -> Flexiv base pose"""
        if self.pose_source != "vive":
            return
        try:
            pose_msg = msg.pose
            ts = msg.header.stamp.to_sec()

            x = pose_msg.position.x
            y = pose_msg.position.y
            z = pose_msg.position.z
            qx = pose_msg.orientation.x
            qy = pose_msg.orientation.y
            qz = pose_msg.orientation.z
            qw = pose_msg.orientation.w

            qpos_vive = [x, y, z, qx, qy, qz, qw]
            gx, gy, gz, gqx, gqy, gqz, gqw = transform_vive_to_gripper(qpos_vive)

            cmd_pose = [
                float(gx),
                float(gy),
                float(gz),
                float(gqw),
                float(gqx),
                float(gqy),
                float(gqz),
            ]
            self._send_cartesian_cmd(cmd_pose, ts, src="vive")
        except Exception as e:
            rospy.logerr("处理 Vive 位姿时出错: %s", e)

    def _send_cartesian_cmd(self, pose_wxyz7, ts: float, src: str = ""):
        """给 Flexiv 发送一条笛卡尔力控命令"""
        now = time.time()
        if now - self.last_arm_cmd_time < self.arm_cmd_interval:
            return
        self.last_arm_cmd_time = now

        try:
            with self.robot_lock:
                # 已经在 main 里切到了 NRT_CARTESIAN_MOTION_FORCE 模式，这里直接发即可
                self.robot.SendCartesianMotionForce(
                    pose_wxyz7,
                    [0.0] * 6,
                    max_linear_vel=self.max_linear_vel,
                    max_angular_vel=self.max_angular_vel,
                )
        except Exception as e:
            rospy.logerr("发送 Cartesian 命令失败(src=%s): %s", src, e)

    # ------------------------------------------------------------------ #
    # 回调：夹爪 clamp
    # ------------------------------------------------------------------ #
    def _clamp_callback(self, msg):
        """
        XVisio clamp 数值 -> Flexiv 夹爪开合比例 [0,1]

        动态解析 AnyMsg:
        - 第一次收到时，根据 connection_header.type 推断具体消息类型
        - 然后反序列化成真正消息，再尝试从几个常见字段里拿数值:
          data / value / clamp / width
        """
        try:
            # 第一次需要从 AnyMsg 中解析出真实类型
            if isinstance(msg, rospy.AnyMsg):
                if self.clamp_msg_class is None:
                    type_str = ""
                    if hasattr(msg, "_connection_header") and msg._connection_header:
                        type_str = msg._connection_header.get("type", "")
                    if type_str:
                        self.clamp_msg_class = get_message_class(type_str)
                        rospy.loginfo("Clamp topic 类型解析为: %s", type_str)
                if self.clamp_msg_class is None:
                    # 还没解析出类型，先跳过
                    return
                real_msg = self.clamp_msg_class()
                real_msg.deserialize(msg._buff)
            else:
                real_msg = msg

            clamp_value = None
            # 尝试几个常见字段名
            for field in ("data", "value", "clamp", "width"):
                if hasattr(real_msg, field):
                    clamp_value = getattr(real_msg, field)
                    break

            if clamp_value is None:
                return

            clamp_value = float(clamp_value)

            # 映射到 [0,1]: 假设 clamp 越小 -> 夹爪越闭合
            ratio = (self.clamp_max_open - clamp_value) / self.clamp_max_open
            ratio = float(np.clip(ratio, 0.0, 1.0))

            now = time.time()
            if self.last_gripper_cmd is not None:
                # 死区 + 频率限制
                if abs(ratio - self.last_gripper_cmd) < self.gripper_deadband:
                    return
                if now - self.last_gripper_cmd_time < self.gripper_cmd_interval:
                    return

            with self.robot_lock:
                # 这里假设你的封装是 open_gripper(0~1)，内部会按比例映射到实际宽度
                self.robot_wrapper.open_gripper(ratio)

            self.last_gripper_cmd = ratio
            self.last_gripper_cmd_time = now
            self.current_gripper_ratio = ratio
        except Exception as e:
            rospy.logerr("处理 clamp 数据时出错: %s", e)


def main():
    parser = argparse.ArgumentParser(
        description="使用 ROS slam/vive/clamp 遥操作 Flexiv 机械臂"
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default="Rizon 4s-063036",
        help="Flexiv 机械臂名称",
    )
    parser.add_argument(
        "--local-name",
        type=str,
        default="Flexiv-GN01",
        help="本地机器人的名称 / 工位 ID",
    )
    parser.add_argument(
        "--xv-serial",
        "-x",
        type=str,
        required=True,
        help="XV 相机序列号，用于 slam/clamp topic，例如 250801DR48FP25002269",
    )
    parser.add_argument(
        "--vive-serial",
        "-v",
        type=str,
        required=True,
        help="Vive 手柄序列号，用于 vive topic，例如 LHR-XXXXXXXX",
    )
    parser.add_argument(
        "--source",
        choices=["slam", "vive"],
        default="vive",
        help="使用 slam 还是 vive 来控制位姿（默认 vive）",
    )
    parser.add_argument(
        "--clamp-max",
        type=float,
        default=88.0,
        help="Clamp 最大开合值（和你采集脚本里保持一致），默认 88",
    )
    parser.add_argument(
        "--max-linear-vel",
        type=float,
        default=0.25,
        help="笛卡尔最大平移速度 (m/s)，默认 0.25",
    )
    parser.add_argument(
        "--max-angular-vel",
        type=float,
        default=1.5,
        help="笛卡尔最大角速度 (rad/s)，默认 1.5",
    )
    parser.add_argument(
        "--home",
        nargs=7,
        type=float,
        default=None,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        help="可选：退出时回到的笛卡尔 home 位姿（和你原来 HOME_TARGET 一样的 7 个数）",
    )

    args = parser.parse_args()

    # 初始化 ROS
    rospy.init_node("flexiv_teleop_from_ros", anonymous=True)

    # 连接 Flexiv
    mode = flexivrdk.Mode
    right_robot = flexiv_robot(args.robot_name, args.local_name)
    rospy.loginfo("已连接 Flexiv: %s (%s)", args.robot_name, args.local_name)

    # 零力传感器 & 切模式
    right_robot.zero_ft_sensor()
    right_robot.switch_mode(mode.NRT_CARTESIAN_MOTION_FORCE)
    rospy.loginfo("已切到 NRT_CARTESIAN_MOTION_FORCE 模式")

    # 打开夹爪（全开）
    right_robot.open_gripper(1.0)
    rospy.loginfo("已打开夹爪，等待 slam/vive/clamp 数据 ...")

    # 创建遥操作控制器
    controller = FlexivRosTeleopController(
        robot_wrapper=right_robot,
        xv_serial=args.xv_serial,
        vive_serial=args.vive_serial,
        pose_source=args.source,
        clamp_max_open=args.clamp_max,
        max_linear_vel=args.max_linear_vel,
        max_angular_vel=args.max_angular_vel,
    )

    rospy.loginfo(
        "遥操作已就绪：位姿来源=%s，按 Ctrl+C 退出程序。",
        args.source,
    )

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("收到 Ctrl+C，准备退出...")
    finally:
        # 可选：回到 home 位姿
        if args.home is not None:
            try:
                rospy.loginfo("回到 home 位姿: %s", args.home)
                right_robot.switch_mode(mode.NRT_CARTESIAN_MOTION_FORCE)
                pose = [
                    float(args.home[0]),
                    float(args.home[1]),
                    float(args.home[2]),
                    float(args.home[3]),
                    float(args.home[4]),
                    float(args.home[5]),
                    float(args.home[6]),
                ]
                right_robot.robot.SendCartesianMotionForce(
                    pose,
                    [0.0] * 6,
                    max_linear_vel=0.04,
                    max_angular_vel=1.5,
                )
            except Exception as e:
                rospy.logwarn("回 home 位姿失败: %s", e)

        rospy.loginfo("程序结束。")


if __name__ == "__main__":
    main()
