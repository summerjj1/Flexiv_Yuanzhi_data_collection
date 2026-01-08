#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python /home/vla/code/data_collection/rollout_flexiv.py \
    --xv-serial 250801DR48FP25002065 \
    --vive-serial LHR-12345678 \
    --source slam \
    --clamp-max 88
"""

import argparse
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List
import pdb
import numpy as np
import rospy
import flexivrdk
from geometry_msgs.msg import PoseStamped
from xv_sdk.msg import PoseStampedConfidence
from roslib.message import get_message_class
from scipy.spatial.transform import Rotation as R

from pose_merge import transform_vive_to_gripper, transform_slam_to_gripper
from api.flexiv_robot import flexiv_robot
import faulthandler, signal, os
faulthandler.register(signal.SIGUSR1)
print("PID:", os.getpid(), flush=True)


def transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local):
    """
    你原来代码里的 transform_to_base_quat，
    这里基本按照你写的计算方式保留（保证行为和 xArm 版本一致）。
    T_base_to_local: 4x4 齐次矩阵
    """
    rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()

    T_local = np.eye(4)
    T_local[:3, :3] = rotation_local
    T_local[:3, 3] = [x, y, z]

    T_base_r = np.matmul(T_local[:3, :3], T_base_to_local[:3, :3])
    x_base, y_base, z_base = T_base_to_local[:3, 3] + T_local[:3, 3]

    rotation_base = R.from_matrix(T_base_r)
    roll_base, pitch_base, yaw_base = rotation_base.as_euler('xyz', degrees=True)
    qx_base, qy_base, qz_base, qw_base = rotation_base.as_quat()

    return (
        x_base,
        y_base,
        z_base,
        qx_base,
        qy_base,
        qz_base,
        qw_base,
        roll_base,
        pitch_base,
        yaw_base,
    )


@dataclass
class PoseSample:
    t: float
    # 原始设备系位姿（x, y, z, qx, qy, qz, qw）
    raw_xyz_quat: np.ndarray          # shape (7,)
    # transform_*_to_gripper 之后的“local/gripper系”位姿（还没到机器人 base）
    local_xyz_quat: np.ndarray = field(default_factory=lambda: np.zeros(7))


@dataclass
class ClampSample:
    t: float
    raw_value: float
    ratio: float   # 已映射到 [0, 1]


@dataclass
class TeleopLogEntry:
    t: float
    # 控制来源: "slam" / "vive"
    source: str
    # 原始位姿 & local 位姿
    raw_xyz_quat: np.ndarray      # shape (7,)
    local_xyz_quat: np.ndarray    # shape (7,)
    # 最终发送给机器人 base 的 pose（x,y,z,qx,qy,qz,qw）
    cmd_pose: np.ndarray          # shape (7,)
    # 夹爪部分
    clamp_raw: float
    clamp_ratio: float

class FlexivRosTeleop:
    def __init__(
        self,
        robot_name: str,
        local_name: str,
        xv_serial: str,
        vive_serial: str,
        pose_source: str = "slam",
        clamp_max_open: float = 88.0,
        max_linear_vel: float = 1,
        max_angular_vel: float = 1.5,
        control_rate: float = 100.0,
    ):
        assert pose_source in ("slam", "vive")
        self.pose_source = pose_source
        self.xv_serial = xv_serial
        self.vive_serial = vive_serial
        self.clamp_max_open = float(clamp_max_open)
        self.max_linear_vel = float(max_linear_vel)
        self.max_angular_vel = float(max_angular_vel)
        self.control_rate = control_rate

        # ---- 连接 Flexiv ----
        self.mode = flexivrdk.Mode
        self.robot_wrapper = flexiv_robot(robot_name, local_name)
        self.robot = self.robot_wrapper.robot

        rospy.loginfo("已连接 Flexiv: %s (%s)", robot_name, local_name)
        self.robot_wrapper.zero_ft_sensor()
        self.robot_wrapper.switch_mode(self.mode.NRT_CARTESIAN_MOTION_FORCE)
        rospy.loginfo("已切到 NRT_CARTESIAN_MOTION_FORCE 模式")

        # 默认把夹爪打开
        self.robot_wrapper.open_gripper(1.0)

        current_states = self.robot.states()
        tcp_pose = current_states.tcp_pose  # [x, y, z, qw, qx, qy, qz]
        base_x, base_y, base_z = tcp_pose[:3]
        qw, qx, qy, qz = tcp_pose[3:]
        # SciPy 的 from_quat 接受 [x, y, z, w]
        rotation_base_to_local = R.from_quat([qx, qy, qz, qw]).as_matrix()

        T_base_to_local = np.eye(4)
        T_base_to_local[:3, :3] = rotation_base_to_local
        T_base_to_local[:3, 3] = [base_x, base_y, base_z]
        self.T_base_to_local = T_base_to_local

        rospy.loginfo(
            "T_base_to_local 已用当前 tcp_pose 初始化: "
            "t = (%.3f, %.3f, %.3f), quat(wxyz) = (%.3f, %.3f, %.3f, %.3f)",
            base_x, base_y, base_z, qw, qx, qy, qz
        )

        # ---- ROS topic ----
        self.slam_topic = f"/xv_sdk/{self.xv_serial}/slam/pose"
        vive_serial_safe = self.vive_serial.replace("-", "_")
        self.vive_topic = f"/vive/{vive_serial_safe}/pose"
        self.clamp_topic = f"/xv_sdk/{self.xv_serial}/clamp/Data"

        rospy.loginfo("slam topic  : %s", self.slam_topic)
        rospy.loginfo("vive topic  : %s", self.vive_topic)
        rospy.loginfo("clamp topic : %s", self.clamp_topic)
        rospy.loginfo("pose source : %s", self.pose_source)

        self._lock = threading.Lock()
        self._last_slam: Optional[PoseSample] = None
        self._last_vive: Optional[PoseSample] = None
        self._last_clamp: Optional[ClampSample] = None

        # clamp 动态解码需要
        self._clamp_msg_class = None

        # 频率限制
        self._last_arm_cmd_time = 0.0
        self._arm_cmd_interval = 1.0 / self.control_rate

        self._last_gripper_cmd = None
        self._last_gripper_cmd_time = 0.0
        self._gripper_deadband = 0.02       # 2% 以内不动
        self._gripper_cmd_interval = 0.01   # 最多 20 Hz

        self.log: List[TeleopLogEntry] = []

        # 订阅 topic
        self._subscribe_topics()


    def _subscribe_topics(self):
        self.slam_sub = rospy.Subscriber(
            self.slam_topic,
            PoseStampedConfidence,
            self._slam_callback,
            queue_size=1,
            buff_size=2**20,
            tcp_nodelay=True,
        )
        self.vive_sub = rospy.Subscriber(
            self.vive_topic,
            PoseStamped,
            self._vive_callback,
            queue_size=1,
            buff_size=2**20,
            tcp_nodelay=True,
        )
        self.clamp_sub = rospy.Subscriber(
            self.clamp_topic,
            rospy.AnyMsg,
            self._clamp_callback,
            queue_size=1,
            buff_size=2**20,
            tcp_nodelay=True,
        )

    def _slam_callback(self, msg: PoseStampedConfidence):
        try:
            pose_msg = msg.poseMsg.pose
            t = msg.poseMsg.header.stamp.to_sec()

            raw = np.array(
                [
                    pose_msg.position.x,
                    pose_msg.position.y,
                    pose_msg.position.z,
                    pose_msg.orientation.x,
                    pose_msg.orientation.y,
                    pose_msg.orientation.z,
                    pose_msg.orientation.w,
                ],
                dtype=float,
            )

            gx, gy, gz, gqx, gqy, gqz, gqw = transform_slam_to_gripper(raw.tolist())
            local = np.array([gx, gy, gz, gqx, gqy, gqz, gqw], dtype=float)

            sample = PoseSample(t=t, raw_xyz_quat=raw, local_xyz_quat=local)
            print("sample", sample)
            with self._lock:
                self._last_slam = sample
        except Exception as e:
            rospy.logerr("slam 回调出错: %s", e)

    def _vive_callback(self, msg: PoseStamped):
        try:
            pose_msg = msg.pose
            t = msg.header.stamp.to_sec()

            raw = np.array(
                [
                    pose_msg.position.x,
                    pose_msg.position.y,
                    pose_msg.position.z,
                    pose_msg.orientation.x,
                    pose_msg.orientation.y,
                    pose_msg.orientation.z,
                    pose_msg.orientation.w,
                ],
                dtype=float,
            )

            gx, gy, gz, gqx, gqy, gqz, gqw = transform_vive_to_gripper(raw.tolist())
            local = np.array([gx, gy, gz, gqx, gqy, gqz, gqw], dtype=float)

            sample = PoseSample(t=t, raw_xyz_quat=raw, local_xyz_quat=local)
            print("------------------------------------------------------------------------------------------------")
            with self._lock:
                self._last_vive = sample
        except Exception as e:
            rospy.logerr("vive 回调出错: %s", e)

    def _clamp_callback(self, msg):
        """
        clamp topic 动态解析:
         - 第一次通过 AnyMsg 的 connection_header.type 找到真实消息类型
         - 然后从 data/value/clamp/width 等字段里拿数值
        """
        try:
            if isinstance(msg, rospy.AnyMsg):
                if self._clamp_msg_class is None:
                    type_str = ""
                    if hasattr(msg, "_connection_header") and msg._connection_header:
                        type_str = msg._connection_header.get("type", "")
                    if type_str:
                        self._clamp_msg_class = get_message_class(type_str)
                        rospy.loginfo("Clamp topic 类型解析为: %s", type_str)

                if self._clamp_msg_class is None:
                    return

                real_msg = self._clamp_msg_class()
                real_msg.deserialize(msg._buff)
            else:
                real_msg = msg

            clamp_value = None
            for field in ("data", "value", "clamp", "width"):
                if hasattr(real_msg, field):
                    clamp_value = getattr(real_msg, field)
                    break
            if clamp_value is None:
                return

            clamp_value = float(clamp_value)
            ratio = (self.clamp_max_open - clamp_value) / self.clamp_max_open
            ratio = float(np.clip(ratio, 0.0, 1.0))

            sample = ClampSample(
                t=rospy.Time.now().to_sec(),
                raw_value=clamp_value,
                ratio=ratio,
            )
            with self._lock:
                self._last_clamp = sample

            print("------------------------------------------------------------------------------------------------")
        except Exception as e:
            rospy.logerr("clamp 回调出错: %s", e)


    def run(self):
        rate = rospy.Rate(self.control_rate)
        rospy.loginfo("开始 Flexiv ROS 遥操作，来源: %s", self.pose_source)

        try:
            while not rospy.is_shutdown():
                self._step()
                rate.sleep()
        except rospy.ROSInterruptException:
            pass
        except KeyboardInterrupt:
            rospy.loginfo("收到 Ctrl+C，退出控制循环")
        print("------------------------------------------------------------------------------------------------")

        rospy.loginfo("控制循环结束，共记录 %d 条 log", len(self.log))

    def _step(self):
        """单次控制循环：取最新 pose/clamp，做坐标系转换 + 下发指令 + 记录日志"""
        with self._lock:
            slam = self._last_slam
            vive = self._last_vive
            clamp = self._last_clamp

        # 选择控制来源
        if self.pose_source == "slam":
            pose_sample = slam
            source_name = "slam"
        else:
            pose_sample = vive
            source_name = "vive"

        if pose_sample is None:
            # 还没有任何位姿
            return

        # 当前时间（用于日志）
        t_now = rospy.Time.now().to_sec()

        local_pose = pose_sample.local_xyz_quat.copy()
        x_l, y_l, z_l, qx_l, qy_l, qz_l, qw_l = local_pose

        (
            x_b,
            y_b,
            z_b,
            qx_b,
            qy_b,
            qz_b,
            qw_b,
            roll_b,
            pitch_b,
            yaw_b,
        ) = transform_to_base_quat(
            x_l,
            y_l,
            z_l,
            qx_l,
            qy_l,
            qz_l,
            qw_l,
            self.T_base_to_local,
        )
        print("------------------------------------------------------------------------------------------------")

        # base 系下的 7D (xyz + quat)
        cmd_pose = np.array(
            [x_b, y_b, z_b, qx_b, qy_b, qz_b, qw_b],
            dtype=float,
        )

        xyz = cmd_pose[:3]
        qx_c, qy_c, qz_c, qw_c = cmd_pose[3:]

        # Flexiv 要求 [x,y,z, qw,qx,qy,qz]
        pose_wxyz7 = np.array(
            [xyz[0], xyz[1], xyz[2], qw_c, qx_c, qy_c, qz_c],
            dtype=float,
        )

        now = time.time()
        if now - self._last_arm_cmd_time >= self._arm_cmd_interval:
            try:
                self._last_arm_cmd_time = now
                self.robot.SendCartesianMotionForce(
                    pose_wxyz7.tolist(),
                    [0.0] * 6,
                    max_linear_vel=self.max_linear_vel,
                    max_angular_vel=self.max_angular_vel,
                )
            except Exception as e:
                rospy.logerr("发送 Cartesian 命令失败: %s", e)

        clamp_raw = 0.0
        clamp_ratio = 1.0
        if clamp is not None:
            clamp_raw = clamp.raw_value
            clamp_ratio = clamp.ratio

            if self._last_gripper_cmd is None:
                need_send = True
            else:
                need_send = (
                    abs(clamp_ratio - self._last_gripper_cmd) >= self._gripper_deadband
                    and now - self._last_gripper_cmd_time >= self._gripper_cmd_interval
                )

            if need_send:
                try:
                    self.robot_wrapper.open_gripper(clamp_ratio)
                    self._last_gripper_cmd = clamp_ratio
                    self._last_gripper_cmd_time = now
                except Exception as e:
                    rospy.logerr("下发夹爪命令失败: %s", e)

        entry = TeleopLogEntry(
            t=t_now,
            source=source_name,
            raw_xyz_quat=pose_sample.raw_xyz_quat.copy(),
            local_xyz_quat=pose_sample.local_xyz_quat.copy(),
            cmd_pose=cmd_pose.copy(),
            clamp_raw=clamp_raw,
            clamp_ratio=clamp_ratio,
        )
        self.log.append(entry)


def main():
    # pdb.set_trace()
    parser = argparse.ArgumentParser(
        description="用 ROS 的 slam/vive/clamp 遥操作 Flexiv 机械臂"
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default="Rizon 4s-063036",
        help="Flexiv 机械臂名称（和 Flexiv Studio 中一致）",
    )
    parser.add_argument(
        "--local-name",
        type=str,
        default="Flexiv-GN01",
        help="本机 / 工位名，用于 flexiv_robot 封装",
    )
    parser.add_argument(
        "--xv-serial",
        "-x",
        type=str,
        required=True,
        help="XV 相机序列号，例如 250801DR48FP25002087",
    )
    parser.add_argument(
        "--vive-serial",
        "-v",
        type=str,
        required=True,
        help="Vive 手柄序列号，例如 LHR-XXXXXXXX",
    )
    parser.add_argument(
        "--source",
        choices=["slam", "vive"],
        default="vive",
        help="选择用 slam 还是 vive 控制位姿",
    )
    parser.add_argument(
        "--clamp-max",
        type=float,
        default=88.0,
        help="Clamp 最大开合值（和你采数据时一致），影响映射到 [0,1]",
    )
    parser.add_argument(
        "--max-linear-vel",
        type=float,
        default=0.25,
        help="笛卡尔最大线速度 m/s",
    )
    parser.add_argument(
        "--max-angular-vel",
        type=float,
        default=1.5,
        help="笛卡尔最大角速度 rad/s",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=100.0,
        help="控制循环频率 Hz（笛卡尔命令下发频率）",
    )
    # pdb.set_trace()
    args = parser.parse_args()

    rospy.init_node("flexiv_teleop_slam_vive_clamp", anonymous=True)
    # pdb.set_trace()
    controller = FlexivRosTeleop(
        robot_name=args.robot_name,
        local_name=args.local_name,
        xv_serial=args.xv_serial,
        vive_serial=args.vive_serial,
        pose_source=args.source,
        clamp_max_open=args.clamp_max,
        max_linear_vel=args.max_linear_vel,
        max_angular_vel=args.max_angular_vel,
        control_rate=args.rate,
    )
    # pdb.set_trace()
    controller.run()


if __name__ == "__main__":
    main()
