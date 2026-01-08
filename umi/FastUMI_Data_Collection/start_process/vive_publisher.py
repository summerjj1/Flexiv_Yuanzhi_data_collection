#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import sys
import argparse
from collections import defaultdict

import numpy as np
import openvr

# ====== ROS1 ======
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped, Vector3Stamped  # NEW
import tf
import tf2_ros


# ============ 3x4 刚体矩阵工具 ============
def mat34_from_openvr(m34) -> np.ndarray:
    return np.array(m34.m, dtype=np.float64)

def mat34(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.zeros((3, 4), dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3]  = t
    return M

def invert34(M: np.ndarray) -> np.ndarray:
    R = M[:3, :3]
    t = M[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return mat34(R_inv, t_inv)

def mul34(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A44 = np.eye(4); A44[:3, :3] = A[:3, :3]; A44[:3, 3] = A[:3, 3]
    B44 = np.eye(4); B44[:3, :3] = B[:3, :3]; B44[:3, 3] = B[:3, 3]
    C44 = A44 @ B44
    return C44[:3, :]

def mat34_to_trans_quat(M: np.ndarray):
    R = M[:3, :3]
    t = M[:3, 3]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    qx, qy, qz, qw = tf.transformations.quaternion_from_matrix(T)
    return t[0], t[1], t[2], qx, qy, qz, qw

def mat34_to_rpy(M: np.ndarray, degrees: bool = False):  # NEW
    """返回 (roll, pitch, yaw)，默认弧度；degrees=True 则为角度。"""
    T = np.eye(4)
    T[:3, :3] = M[:3, :3]
    T[:3,  3] = M[:3,  3]
    # sxyz: 先绕X(roll)，再绕Y(pitch)，再绕Z(yaw)
    roll, pitch, yaw = tf.transformations.euler_from_matrix(T, axes='sxyz')
    if degrees:
        roll, pitch, yaw = map(math.degrees, (roll, pitch, yaw))
    return roll, pitch, yaw


# ============ OpenVR ============
def init_openvr_system_only():
    openvr.init(openvr.VRApplication_Other)  # 无头显可用
    sys_obj = openvr.VRSystem()
    return sys_obj

def list_trackers(sys_obj):
    """返回 {serial_number: device_index} 字典"""
    trackers = {}
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if sys_obj.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_GenericTracker:
            try:
                serial = sys_obj.getStringTrackedDeviceProperty(i, openvr.Prop_SerialNumber_String)
                trackers[serial] = i
            except Exception as e:
                rospy.logwarn(f"无法获取设备 {i} 的序列号: {e}")
                # 如果无法获取序列号，使用索引作为后备
                trackers[f"tracker_{i}"] = i
    return trackers


# ============ 主逻辑 ============
class ViveTrackersPublisher:
    def __init__(self, args):
        self.args = args

        # ROS
        rospy.init_node("vive_trackers_publisher", anonymous=False)
        self.frame_app = args.app_frame
        self.pub_rate  = rospy.Rate(args.rate)

        # FPS（按索引）
        self.fps_time = {}
        self.fps_val  = defaultdict(lambda: 0.0)

        # TF（可选）
        self.br = tf2_ros.TransformBroadcaster() if args.publish_tf else None

        # OpenVR
        self.sys_obj = init_openvr_system_only()

        # 找到所有 tracker（serial -> index）
        self.trackers = list_trackers(self.sys_obj)
        if not self.trackers:
            openvr.shutdown()
            raise RuntimeError("未找到 Vive Tracker。请确认基站开启、Tracker 已配对并在 SteamVR 中可见。")

        rospy.loginfo("发现 Trackers (序列号 -> 索引):")
        for serial, idx in self.trackers.items():
            rospy.loginfo(f"  {serial} -> 设备索引 {idx}")

        # 选择用于标定的 Tracker
        self.calib_serial = None
        self.calib_index = None
        if args.calib_serial is not None:
            if args.calib_serial in self.trackers:
                self.calib_serial = args.calib_serial
                self.calib_index = self.trackers[args.calib_serial]
            else:
                rospy.logwarn(f"指定的序列号 {args.calib_serial} 不存在，使用第一个 tracker。")
        elif args.calib_index is not None:
            for serial, idx in self.trackers.items():
                if idx == args.calib_index:
                    self.calib_serial = serial
                    self.calib_index = idx
                    break
            if self.calib_index is None:
                rospy.logwarn(f"指定的索引 {args.calib_index} 不存在，使用第一个 tracker。")
        if self.calib_serial is None:
            self.calib_serial = list(self.trackers.keys())[0]
            self.calib_index = self.trackers[self.calib_serial]
        rospy.loginfo(f"用于标定的 Tracker: {self.calib_serial} (设备索引 {self.calib_index})")

        # FULL_6DOF 标定：app = 标定瞬间该 tracker 的位姿
        self.AppFromWorld = self._do_full6dof_calibration()

        # ******** 新增：为每个 tracker 记录“app 框架下的基线姿态”（用于各自归零） ********
        self.app_baseline = {}
        rospy.loginfo("为每个 tracker 记录基线（app 框架下）以实现各自归零…")
        time.sleep(args.calib_wait)  # 稍等稳定
        poses = self.sys_obj.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0.0, openvr.k_unMaxTrackedDeviceCount
        )
        for serial, idx in self.trackers.items():
            p = poses[idx]
            if p.bPoseIsValid:
                T_world_tracker = mat34_from_openvr(p.mDeviceToAbsoluteTracking)
                T_app_tracker   = mul34(self.AppFromWorld, T_world_tracker)
                self.app_baseline[serial] = T_app_tracker
            else:
                rospy.logwarn(f"tracker {serial} 基线无效，使用单位阵（不做额外归零）")
                self.app_baseline[serial] = mat34(np.eye(3), np.zeros(3))
        # ******** 新增结束 ********

        # 建立 ROS publishers
        self.pose_pubs = {}
        self.rpy_pubs  = {}
        for serial in self.trackers.keys():
            topic_serial = serial.replace('-', '_')
            topic_pose = f"/vive/{topic_serial}/pose"
            topic_rpy  = f"/vive/{topic_serial}/rpy"
            self.pose_pubs[serial] = rospy.Publisher(topic_pose, PoseStamped, queue_size=10)
            self.rpy_pubs[serial]  = rospy.Publisher(topic_rpy,  Vector3Stamped, queue_size=10)
            rospy.loginfo(f"发布 tracker {serial} 到 {topic_pose} 和 {topic_rpy}")

    def _do_full6dof_calibration(self) -> np.ndarray:
        rospy.loginfo("请将用于标定的 Tracker 放到“应用原点”的位置与朝向，保持稳定中……")
        time.sleep(self.args.calib_wait)
        while not rospy.is_shutdown():
            poses = self.sys_obj.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0.0, openvr.k_unMaxTrackedDeviceCount
            )
            p = poses[self.calib_index]
            if p.bPoseIsValid:
                T_world_tracker0 = mat34_from_openvr(p.mDeviceToAbsoluteTracking)
                rospy.loginfo("✅ 标定完成（FULL_6DOF）: app 原点 = 标定瞬间该 Tracker 的位姿")
                return invert34(T_world_tracker0)
            rospy.logwarn_throttle(2.0, "标定 Tracker 暂无有效姿态，重试中……")
            time.sleep(0.01)

    def _publish_pose_and_rpy(self, serial: str, M_app_tracker: np.ndarray):
        # PoseStamped
        tx, ty, tz, qx, qy, qz, qw = mat34_to_trans_quat(M_app_tracker)
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_app
        msg.pose.position.x = tx
        msg.pose.position.y = ty
        msg.pose.position.z = tz
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pose_pubs[serial].publish(msg)

        # Vector3Stamped (roll, pitch, yaw)
        r, p, y = mat34_to_rpy(M_app_tracker, degrees=self.args.rpy_deg)
        v = Vector3Stamped()
        v.header.stamp = msg.header.stamp
        v.header.frame_id = self.frame_app
        v.vector.x = r
        v.vector.y = p
        v.vector.z = y
        self.rpy_pubs[serial].publish(v)

        # TF（可选）
        if self.br is not None:
            tfm = TransformStamped()
            tfm.header.stamp = msg.header.stamp
            tfm.header.frame_id = self.frame_app
            topic_serial = serial.replace('-', '_')
            tfm.child_frame_id  = f"vive_{topic_serial}"
            tfm.transform.translation.x = tx
            tfm.transform.translation.y = ty
            tfm.transform.translation.z = tz
            tfm.transform.rotation.x = qx
            tfm.transform.rotation.y = qy
            tfm.transform.rotation.z = qz
            tfm.transform.rotation.w = qw
            self.br.sendTransform(tfm)

    def spin(self):
        rospy.loginfo("开始循环：读取 OpenVR → 变换到 app → 各自归零 → 发布 ROS PoseStamped + RPY（可选 TF）…")
        while not rospy.is_shutdown():
            poses = self.sys_obj.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0.0, openvr.k_unMaxTrackedDeviceCount
            )

            for serial, idx in self.trackers.items():
                pose = poses[idx]
                if not pose.bPoseIsValid:
                    rospy.logwarn_throttle(2.0, f"tracker {serial} 姿态无效（可能看不到基站）…")
                    continue

                # 世界 → app
                T_world_tracker = mat34_from_openvr(pose.mDeviceToAbsoluteTracking)
                T_app_tracker   = mul34(self.AppFromWorld, T_world_tracker)

                B_app = self.app_baseline.get(serial)
                if B_app is not None:
                    T_app_rel = mul34(invert34(B_app), T_app_tracker)
                else:
                    T_app_rel = T_app_tracker

                # FPS
                now_t = time.perf_counter()
                if serial in self.fps_time:
                    dt = now_t - self.fps_time[serial]
                    if dt > 0:
                        self.fps_val[serial] = 1.0 / dt
                self.fps_time[serial] = now_t

                # 发布 Pose + RPY（改为发布 T_app_rel）
                self._publish_pose_and_rpy(serial, T_app_rel)

                # 终端打印
                if self.args.print_status:
                    pvec = T_app_rel[:3, 3]
                    Rm   = T_app_rel[:3, :3]
                    yaw_deg = math.degrees(math.atan2(Rm[1, 0], Rm[0, 0]))
                    sys.stdout.write(
                        f"\r[{serial}] pos=({pvec[0]:+.3f},{pvec[1]:+.3f},{pvec[2]:+.3f}) "
                        f"yaw~={yaw_deg:+.1f}° | FPS={self.fps_val[serial]:.1f}Hz   "
                    )
                    sys.stdout.flush()

            self.pub_rate.sleep()


def parse_args():
    ap = argparse.ArgumentParser(description="OpenVR Vive Trackers → 应用坐标变换 → ROS 发布（按序列号命名）")
    ap.add_argument("--rate", type=int, default=100, help="发布频率（Hz）")
    ap.add_argument("--app-frame", type=str, default="app", help="应用坐标系 frame_id 名称")
    ap.add_argument("--calib-serial", type=str, default=None, help="用于标定 FULL_6DOF 的 Tracker 序列号（默认用第一个）")
    ap.add_argument("--calib-index", type=int, default=None, help="用于标定 FULL_6DOF 的 Tracker 索引（向后兼容，默认用第一个）")
    ap.add_argument("--calib-wait", type=float, default=1.0, help="标定前/记录基线前静置等待秒数")
    ap.add_argument("--publish-tf", action="store_true", help="是否发布 TF（app -> vive_<serial>）")
    ap.add_argument("--print-status", action="store_true", help="终端打印实时状态")
    ap.add_argument("--rpy-deg", action="store_true", help="将 RPY 以角度发布（默认弧度）")
    return ap.parse_args()


def main():
    args = parse_args()
    try:
        node = ViveTrackersPublisher(args)
        node.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"运行时出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            openvr.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
