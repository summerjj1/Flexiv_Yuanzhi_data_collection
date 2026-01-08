#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import sys
import os
import signal
import argparse
import cv2
import csv
import threading
import time
import numpy as np
import json
import subprocess
import shutil
from collections import deque
from queue import Queue, Empty
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from xv_sdk.msg import PoseStampedConfidence
from datetime import datetime
import enum
import psutil
from roslib.message import get_message_class
from scipy.spatial.transform import Rotation, Slerp
from pose_merge import transform_vive_to_gripper, transform_slam_to_gripper

try:
    cv2.setNumThreads(1)
except Exception:
    pass


class RecordingState(enum.Enum):
    """录制状态枚举"""
    IDLE = "IDLE"           # 空闲状态，等待开始
    RECORDING = "RECORDING" # 录制中
    SAVING = "SAVING"       # 保存中
    FINISHED = "FINISHED"   # 完成


class SingleDeviceRecorder:
    """单设备数据采集器"""

    def __init__(self, device_config, output_dir=None, enable_tof=True, auto_start=False, encode_video=False):
        """
        参数：
            device_config: dict, 设备配置 {'xv_serial': str, 'vive_serial': str, 'label': str}
            output_dir: str, 输出目录
            enable_tof: bool, 是否启用ToF
            auto_start: bool, 是否在初始化时自动等待用户输入
            encode_video: bool, 会话结束是否合成MP4（由RAW直出）
        """
        # 解析设备配置
        self.xv_serial = device_config['xv_serial']
        self.vive_serial = device_config['vive_serial']
        self.device_label = device_config.get('label', '')

        self.device_serial = self.xv_serial

        self.running = True
        self.bridge = CvBridge()
        self.enable_tof = enable_tof
        self.auto_start = auto_start
        self.encode_video = encode_video
        self.global_start_time = None

        # 输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = f"{self.device_label}_{self.xv_serial}" if self.device_label else self.xv_serial
            self.output_dir = os.path.join("DATA", base_dir, f"session_{timestamp}")
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 状态
        self.recording_state = RecordingState.IDLE
        self.recording_start_time = None
        self.recording_end_time = None

        # Topics
        self.rgb_topic = f"/xv_sdk/{self.xv_serial}/color_camera/image"
        self.slam_topic = f"/xv_sdk/{self.xv_serial}/slam/pose"
        vive_serial_safe = self.vive_serial.replace('-', '_')
        self.vive_topic = f"/vive/{vive_serial_safe}/pose"
        self.tof_topic = f"/xv_sdk/{self.xv_serial}/tof_camera/point_cloud"
        self.clamp_topic = f"/xv_sdk/{self.xv_serial}/clamp/Data"

        # 计数与统计
        self.rgb_count = 0
        self.slam_count = 0
        self.vive_count = 0
        self.tof_count = 0
        self.clamp_count = 0
        self.merged_count = 0

        # RGB 写入
        self.frame_width = None
        self.frame_height = None
        self.first_rgb_frame = False
        self.timestamp_writer = None
        self.timestamp_file = None

        # RAW 写入相关
        self.rgb_queue = Queue(maxsize=200)
        self.rgb_writer_thread = None
        self._stop_writers = threading.Event()
        self.rgb_frame_index = 0
        self.rgb_raw_file = None
        self.rgb_raw_path = None
        self.rgb_meta_written = False

        # RGB FPS 估计
        self.rgb_timestamps = deque(maxlen=300)
        self.rgb_current_fps = 0.0
        self.rgb_avg_fps = 0.0
        self.rgb_fps_history = deque(maxlen=300)
        self.rgb_last_fps_calc_time = time.time()
        self.fps_calc_interval = 1

        # RGB 对齐
        self.rgb_first_timestamp = None
        self.rgb_time_offset = 0.0
        self.rgb_offset_ready = False

        # SLAM/VIVE
        self.slam_buffer = deque(maxlen=30000)  # 500Hz ≈ 6s
        self.vive_buffer = deque(maxlen=10000)  # 100Hz ≈ 10s

        self.slam_file_handle = None
        self.vive_file_handle = None
        self.clamp_file_handle = None
        self.tof_timestamp_writer = None
        self.tof_timestamp_file = None

        # Vive 对齐状态
        self.slam_first_timestamp = None
        self.vive_first_timestamp = None
        self.vive_time_offset = None
        self.offset_ready = False
        self.vive_pending_buffer = []

        # 速度估计
        self.slam_prev_pose = None
        self.slam_prev_timestamp = None
        self.slam_linear_velocity = 0.0
        self.slam_angular_velocity = 0.0
        self.slam_velocity_history = deque(maxlen=10)

        self.vive_prev_pose = None
        self.vive_prev_timestamp = None
        self.vive_linear_velocity = 0.0
        self.vive_angular_velocity = 0.0
        self.vive_velocity_history = deque(maxlen=10)

        # 融合配置/统计
        self.velocity_threshold = 1.0
        self.time_match_threshold = 0.01
        self.merge_stats = {
            'total_merged': 0,
            'averaged': 0,
            'use_slam': 0,
            'use_vive': 0,
            'both_high': 0,
            'skipped': 0,
            'no_match': 0
        }
        # SLAM VIVE ToF FPS 估计
        self.slam_timestamps = deque(maxlen=300)
        self.slam_current_fps = 0.0
        self.slam_last_fps_calc_time = time.time()

        self.vive_timestamps = deque(maxlen=300)
        self.vive_current_fps = 0.0
        self.vive_last_fps_calc_time = time.time()

        self.tof_timestamps = deque(maxlen=300)
        self.tof_current_fps = 0.0
        self.tof_last_fps_calc_time = time.time()

        # ToF阻塞队列写盘线程
        self.tof_queue = Queue(maxsize=400)
        self.tof_writer_thread = None
        self._tof_index = 0
        self.tof_stats = {
            'min_depth': float('inf'),
            'max_depth': 0,
            'total_points': 0,
            'total_invalid_points': 0
        }

        # Clamp 动态类型解析
        self.clamp_msg_class = None

        # 初始化 ROS
        if not rospy.core.is_initialized():
            rospy.init_node('single_session_data_recorder', anonymous=True)

        # 信号处理
        signal.signal(signal.SIGINT, self.signal_handler)

        # 信息
        if not self.auto_start:
            print("=" * 60)
            print("摄像头 单设备数据采集器")
            print("=" * 60)
        if self.device_label:
            print(f"设备标签: {self.device_label}")
        print(f"XV 序列号: {self.xv_serial}")
        print(f"Vive 序列号: {self.vive_serial}")
        print(f"输出目录: {self.output_dir}")
        if not self.auto_start:
            print("=" * 60)

        # Topic 检查与订阅
        self.check_topics()
        self.subscribe_topics()

        if not self.auto_start:
            print("\n按回车键开始录制...")
            input()
            self.start_recording()
        else:
            print(f"[{self.device_label}] 已初始化，等待启动...")

    def _closest_slam_ts(self, target_ts):
        if not self.slam_buffer:
            return None
        return min(self.slam_buffer, key=lambda x: abs(x[1] - target_ts))[1]

    def parse_tum_line(self, tum_line):
        """解析TUM行为 [x y z qx qy qz qw]"""
        try:
            parts = tum_line.strip().split()
            if len(parts) < 8:
                return None
            pose = [float(parts[i]) for i in range(1, 8)]
            return pose
        except Exception:
            return None

    def calculate_linear_velocity(self, current_pose, prev_pose, dt):
        if prev_pose is None or dt <= 0:
            return 0.0
        p_curr = np.array(current_pose[:3])
        p_prev = np.array(prev_pose[:3])
        return float(np.linalg.norm(p_curr - p_prev) / dt)

    def calculate_angular_velocity(self, current_pose, prev_pose, dt):
        if prev_pose is None or dt <= 0:
            return 0.0
        try:
            q_curr = Rotation.from_quat(current_pose[3:])
            q_prev = Rotation.from_quat(prev_pose[3:])
            q_diff = q_curr * q_prev.inv()
            angle = np.linalg.norm(q_diff.as_rotvec())
            return float(angle / dt)
        except Exception:
            return 0.0

    def smooth_velocity(self, velocity_history):
        if len(velocity_history) == 0:
            return 0.0
        return float(sum(velocity_history) / len(velocity_history))
    
    def _safe_flush_and_fsync(self, fh):
        try:
            if fh and (not fh.closed):
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:
            pass

    def flush_all_stream_files(self):
        try:
            self.process_pending_vive_data()
        except Exception:
            pass
        self._safe_flush_and_fsync(self.timestamp_file)
        self._safe_flush_and_fsync(self.slam_file_handle)
        self._safe_flush_and_fsync(self.vive_file_handle)
        self._safe_flush_and_fsync(self.clamp_file_handle)
        self._safe_flush_and_fsync(self.tof_timestamp_file)

    def average_poses(self, pose1, pose2):
        """
        融合两个pose（位置取平均，四元数使用SLERP）
        pose: [x, y, z, qx, qy, qz, qw]
        """
        try:
            # 位置直接算均值
            pos1 = np.asarray(pose1[:3], dtype=float)
            pos2 = np.asarray(pose2[:3], dtype=float)
            pos_avg = (pos1 + pos2) / 2.0

            # 构造两帧 quaternion 序列
            q_arr = np.array([
                pose1[3:7],  # q1: [qx, qy, qz, qw]
                pose2[3:7],  # q2
            ], dtype=float)

            key_times = [0.0, 1.0]
            key_rots = Rotation.from_quat(q_arr)  # 这里包含两帧
            slerp = Slerp(key_times, key_rots)
            q_avg = slerp(0.5).as_quat()

            averaged_pose = pos_avg.tolist() + q_avg.tolist()
            return averaged_pose

        except Exception as e:
            print(f"警告: SLERP失败，使用降级方案: {e}")
            pos_avg = [(pose1[i] + pose2[i]) / 2.0 for i in range(3)]
            q1 = np.asarray(pose1[3:7], dtype=float)
            q2 = np.asarray(pose2[3:7], dtype=float)
            if np.dot(q1, q2) < 0.0:
                q2 = -q2
            q_avg = q1 + q2
            norm = np.linalg.norm(q_avg)
            if norm > 1e-8:
                q_avg /= norm
            else:
                q_avg = q1
            return pos_avg + q_avg.tolist()

    def merge_poses(self, slam_pose, vive_pose, slam_vel, vive_vel):
        th = self.velocity_threshold
        if slam_vel < th and vive_vel < th:
            merged = self.average_poses(slam_pose, vive_pose)
            self.merge_stats['averaged'] += 1
            return merged, "high", "averaged"
        elif slam_vel >= th and vive_vel < th:
            self.merge_stats['use_vive'] += 1
            return vive_pose, "medium", "use_vive"
        elif slam_vel < th and vive_vel >= th:
            self.merge_stats['use_slam'] += 1
            return slam_pose, "medium", "use_slam"
        else:
            self.merge_stats['both_high'] += 1
            return None, None, None

    def check_topics(self):
        try:
            topics = rospy.get_published_topics()
            topic_names = [t[0] for t in topics]

            missing = []
            if self.rgb_topic not in topic_names:
                missing.append(self.rgb_topic)
            if self.slam_topic not in topic_names:
                missing.append(self.slam_topic)
            if self.enable_tof and self.tof_topic not in topic_names:
                missing.append(self.tof_topic)
            if self.vive_topic not in topic_names:
                missing.append(self.vive_topic)
            if self.clamp_topic not in topic_names:
                print(f"警告: 未发现 Clamp 话题: {self.clamp_topic}（将继续运行）")

            if missing:
                print("错误: 以下话题不存在:")
                for t in missing: print("  -", t)
                print("请确认设备与节点已启动且序列号正确")
                sys.exit(1)

            print("Topic 检查通过:")
            print("  RGB:", self.rgb_topic)
            print("  SLAM:", self.slam_topic)
            print("  ToF:", self.tof_topic if self.enable_tof else "已禁用")
            print("  Vive:", self.vive_topic)
            print("  Clamp:", self.clamp_topic)

        except Exception as e:
            print(f"错误: 无法检查话题状态: {e}")
            sys.exit(1)

    def subscribe_topics(self):
        """订阅所有数据话题"""
        from geometry_msgs.msg import PoseStamped

        self.rgb_subscriber = rospy.Subscriber(
            self.rgb_topic, Image, self.rgb_callback,
            queue_size=120, buff_size=2**26, tcp_nodelay=True
        )
        self.slam_subscriber = rospy.Subscriber(
            self.slam_topic, PoseStampedConfidence, self.slam_callback,
            queue_size=200, buff_size=2**20, tcp_nodelay=True
        )
        if self.enable_tof:
            self.tof_subscriber = rospy.Subscriber(
                self.tof_topic, PointCloud2, self.tof_callback,
                queue_size=60, buff_size=2**24, tcp_nodelay=True
            )
        self.vive_subscriber = rospy.Subscriber(
            self.vive_topic, PoseStamped, self.vive_callback,
            queue_size=100, buff_size=2**20, tcp_nodelay=True
        )
        self.clamp_subscriber = rospy.Subscriber(
            self.clamp_topic, rospy.AnyMsg, self.clamp_callback,
            queue_size=100, buff_size=2**20, tcp_nodelay=True
        )

    def prepare_recording(self):
        if self.recording_state != RecordingState.IDLE:
            return
        if self.device_label:
            print(f"[{self.device_label}] 准备录制...")

        self.init_files()

        # 计数与状态复位
        self.rgb_count = self.slam_count = self.vive_count = 0
        self.tof_count = self.clamp_count = self.merged_count = 0

        self.tof_stats = {
            'min_depth': float('inf'),
            'max_depth': 0,
            'total_points': 0,
            'total_invalid_points': 0
        }
        self.merge_stats = {
            'total_merged': 0,
            'averaged': 0,
            'use_slam': 0,
            'use_vive': 0,
            'both_high': 0,
            'skipped': 0,
            'no_match': 0
        }

        # 清对齐状态
        self.slam_buffer.clear()
        self.vive_buffer.clear()
        self.vive_pending_buffer.clear()
        self.slam_prev_pose = None
        self.slam_prev_timestamp = None
        self.slam_linear_velocity = 0.0
        self.slam_angular_velocity = 0.0
        self.slam_velocity_history.clear()
        self.vive_prev_pose = None
        self.vive_prev_timestamp = None
        self.vive_linear_velocity = 0.0
        self.vive_angular_velocity = 0.0
        self.vive_velocity_history.clear()
        self.slam_first_timestamp = None
        self.vive_first_timestamp = None
        self.vive_time_offset = None
        self.offset_ready = False
        self.rgb_first_timestamp = None
        self.rgb_time_offset = 0.0
        self.rgb_offset_ready = False
        self.first_rgb_frame = False
        self.rgb_meta_written = False

        # 启动写盘线程（RAW）
        self._stop_writers.clear()
        self.start_rgb_writer_raw()
        if self.enable_tof:
            self.start_tof_writer()

        if self.device_label:
            print(f"[{self.device_label}] 准备完成")

    def start_recording(self, global_start_time=None):
        if self.recording_state != RecordingState.IDLE:
            return
        if not hasattr(self, 'slam_file_handle') or self.slam_file_handle is None:
            if not self.auto_start:
                print("\n" + "="*60)
                print("开始录制...")
            self.prepare_recording()

        self.recording_state = RecordingState.RECORDING
        self.global_start_time = global_start_time
        self.recording_start_time = time.time()
        if self.global_start_time is not None:
            delta_ms = (self.recording_start_time - self.global_start_time) * 1000.0
            prefix = f"[{self.device_label}] " if self.device_label else ""
            print(f"{prefix}同步开始目标: {self.global_start_time:.3f}s, 实际: {self.recording_start_time:.3f}s, 偏差: {delta_ms:.1f} ms")

        if self.device_label:
            print(f"[{self.device_label}] 开始录制...")
        else:
            print("状态: 录制中...")

        if not self.auto_start:
            print("按回车键停止录制并退出...")
            print("="*60)
            input()
            self.stop_recording()

    def freeze_recording(self):
        if self.recording_state != RecordingState.RECORDING:
            return
        self.recording_end_time = time.time()
        self.recording_state = RecordingState.SAVING

    def stop_recording(self):
        if self.recording_state != RecordingState.RECORDING:
            return
        tail_grace_s = 0.4  
        time.sleep(tail_grace_s)

        # 进入保存阶段
        self.freeze_recording()
        print("停止录制，等待写盘完成...")

        self.save_all_data()
        self.display_summary()

        self.recording_state = RecordingState.FINISHED
        self.running = False

        print("\n录制完成！程序即将退出...")
        print("="*60)

    def run(self):
        try:
            while not rospy.is_shutdown() and self.running:
                rospy.sleep(0.1)
        except rospy.ROSInterruptException:
            pass
        except KeyboardInterrupt:
            print("\n收到中断信号...")
        finally:
            self.cleanup()

    def signal_handler(self, signum, frame):
        print("\n收到中断信号...")
        self.running = False

    def init_files(self):
        try:
            print(f"初始化文件，目录: {self.output_dir}")

            # RGB 目录与时间戳 CSV
            rgb_dir = os.path.join(self.output_dir, "RGB_Images")
            os.makedirs(rgb_dir, exist_ok=True)
            timestamp_path = os.path.join(rgb_dir, "timestamps.csv")
            self.timestamp_file = open(timestamp_path, 'w', newline='')
            self.timestamp_writer = csv.writer(self.timestamp_file)
            self.timestamp_writer.writerow(['frame_index','seq','header_stamp','aligned_stamp'])
            print(f"  RGB时间戳文件创建: {timestamp_path}")

            # SLAM
            slam_dir = os.path.join(self.output_dir, "SLAM_Poses")
            os.makedirs(slam_dir, exist_ok=True)
            slam_file_path = os.path.join(slam_dir, "slam_raw.txt")
            self.slam_file_handle = open(slam_file_path, 'w')
            print(f"  SLAM文件创建: {slam_file_path}")

            # ToF
            if self.enable_tof:
                tof_dir = os.path.join(self.output_dir, "ToF_PointClouds")
                os.makedirs(tof_dir, exist_ok=True)
                tof_timestamp_path = os.path.join(tof_dir, "timestamps.csv")
                self.tof_timestamp_file = open(tof_timestamp_path, 'w', newline='')
                self.tof_timestamp_writer = csv.writer(self.tof_timestamp_file)
                self.tof_timestamp_writer.writerow(['pointcloud_index','timestamp'])
                print(f"  ToF时间戳文件创建: {tof_timestamp_path}")

            # Clamp
            clamp_dir = os.path.join(self.output_dir, "Clamp_Data")
            os.makedirs(clamp_dir, exist_ok=True)
            clamp_file_path = os.path.join(clamp_dir, "clamp_data_tum.txt")
            self.clamp_file_handle = open(clamp_file_path, 'w')
            print(f"  Clamp TUM文件创建: {clamp_file_path}")

            # Vive
            vive_dir = os.path.join(self.output_dir, "Vive_Poses")
            os.makedirs(vive_dir, exist_ok=True)
            vive_file_path = os.path.join(vive_dir, "vive_data_tum.txt")
            self.vive_file_handle = open(vive_file_path, 'w')
            print(f"  Vive TUM文件创建: {vive_file_path}")

            # Merged 目录
            merged_dir = os.path.join(self.output_dir, "Merged_Trajectory")
            os.makedirs(merged_dir, exist_ok=True)

            print("所有文件初始化完成")
        except Exception as e:
            print(f"错误: 初始化文件失败: {e}")
            self.recording_state = RecordingState.IDLE

    def cleanup_files(self):
        try:
            if self.timestamp_file:
                self.timestamp_file.flush(); self.timestamp_file.close()
                self.timestamp_file = None
                print("  RGB时间戳文件已关闭")
            if self.slam_file_handle:
                self.slam_file_handle.flush(); self.slam_file_handle.close()
                self.slam_file_handle = None
                print("  SLAM文件已关闭")
            if self.tof_timestamp_file:
                self.tof_timestamp_file.flush(); self.tof_timestamp_file.close()
                self.tof_timestamp_file = None
                print("  ToF时间戳文件已关闭")
            if self.clamp_file_handle:
                self.clamp_file_handle.flush(); self.clamp_file_handle.close()
                self.clamp_file_handle = None
                print("  Clamp文件已关闭")
            if self.vive_file_handle:
                self.vive_file_handle.flush(); self.vive_file_handle.close()
                self.vive_file_handle = None
                print("  Vive文件已关闭")
            if getattr(self, "rgb_raw_file", None):
                try:
                    self.rgb_raw_file.flush(); self.rgb_raw_file.close()
                    print("  RGB RAW 文件已关闭")
                except Exception:
                    pass
                self.rgb_raw_file = None
        except Exception as e:
            print(f"错误: 清理文件时出错: {e}")

    def cleanup(self):
        print("\n正在清理资源...")
        if self.recording_state == RecordingState.RECORDING:
            self.stop_recording()

        # 先确保所有数据写完
        if self.recording_state != RecordingState.FINISHED:
            self.save_all_data()

        # 停止后台线程
        try:
            self._stop_writers.set()
            try:
                self.rgb_queue.put(None)
            except Exception:
                pass
            if self.enable_tof:
                try:
                    self.tof_queue.put(None)
                except Exception:
                    pass
            if self.rgb_writer_thread:
                self.rgb_writer_thread.join(timeout=5)
            if self.tof_writer_thread:
                self.tof_writer_thread.join(timeout=5)
        except Exception:
            pass

        self.cleanup_files()

        if self.encode_video:
            self.encode_raw_to_mp4(framerate=60)

        print("="*60)
        print("数据采集完成!")
        print(f"数据已保存到: {self.output_dir}")
        print("="*60)

    def start_rgb_writer_raw(self):
        """队列消费：写 RAW(BGR8 连续字节) + 写时间戳 CSV"""
        rgb_dir = os.path.join(self.output_dir, "RGB_Images")
        os.makedirs(rgb_dir, exist_ok=True)
        self.rgb_raw_path = os.path.join(rgb_dir, "frames.bgr8")
        self.rgb_raw_file = open(self.rgb_raw_path, "wb", buffering=64 * 1024 * 1024)

        def _loop():
            while not self._stop_writers.is_set():
                try:
                    item = self.rgb_queue.get(timeout=0.2)
                except Empty:
                    continue

                if item is None:
                    break

                frame_index, frame_bytes, raw_ts, aligned_ts, seq = item
                try:
                    self.rgb_raw_file.write(frame_bytes)
                    if self.timestamp_writer and self.timestamp_file and not self.timestamp_file.closed:
                        self.timestamp_writer.writerow([
                            int(frame_index), int(seq),
                            f"{raw_ts:.9f}", f"{aligned_ts:.9f}"
                        ])
                    self.rgb_count += 1
                except Exception as e:
                    if self.running:
                        print(f"错误: 写入RGB RAW时出错: {e}")
                finally:
                    self.rgb_queue.task_done()

        self.rgb_writer_thread = threading.Thread(target=_loop, daemon=True)
        self.rgb_writer_thread.start()

    def start_tof_writer(self):
        tof_dir = os.path.join(self.output_dir, "ToF_PointClouds")
        os.makedirs(os.path.join(tof_dir, "PointClouds"), exist_ok=True)

        def _loop():
            while not self._stop_writers.is_set():
                try:
                    item = self.tof_queue.get(timeout=0.2)
                except Empty:
                    continue
                if item is None:
                    break
                msg, ts = item
                pointcloud_path = os.path.join(tof_dir, "PointClouds", f"pointcloud_{self._tof_index:06d}.pcd")
                self.save_pointcloud_to_pcd(msg, pointcloud_path)
                if self.tof_timestamp_writer:
                    self.tof_timestamp_writer.writerow([self._tof_index, f"{ts:.9f}"])
                self._tof_index += 1
                self.tof_queue.task_done()

        self.tof_writer_thread = threading.Thread(target=_loop, daemon=True)
        self.tof_writer_thread.start()

    def rgb_callback(self, msg: Image):
        if self.recording_state != RecordingState.RECORDING:
            return
        try:
            rgb_ts = msg.header.stamp.to_sec()
            seq = getattr(msg.header, 'seq', -1)

            # 首帧对齐：用第一帧 RGB 对齐到最近的 SLAM 时间戳
            if self.rgb_first_timestamp is None:
                self.rgb_first_timestamp = rgb_ts
                slam_ts0 = self._closest_slam_ts(self.rgb_first_timestamp)
                if slam_ts0 is not None:
                    self.rgb_time_offset = slam_ts0 - self.rgb_first_timestamp
                    self.rgb_offset_ready = True
                    print(f"✓ RGB 时间戳对齐就绪")

            ts_aligned = rgb_ts + (self.rgb_time_offset if self.rgb_offset_ready else 0.0)

            # FPS 估计
            if (self.rgb_count % self.fps_calc_interval) == 0:
                self.calculate_rgb_fps(ts_aligned)

            # 使用 ROS Image 原始字节
            h, w = msg.height, msg.width
            enc = getattr(msg, 'encoding', 'bgr8')

            if enc == 'bgr8':
                frame_bytes = bytes(msg.data)
            elif enc == 'rgb8':
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
                bgr = arr[..., ::-1].copy()
                frame_bytes = bgr.tobytes()
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                frame_bytes = cv_image.tobytes()

            if not self.first_rgb_frame:
                self.frame_height, self.frame_width = int(h), int(w)
                self.first_rgb_frame = True
            if not self.rgb_meta_written and self.first_rgb_frame:
                meta = {
                    "width": int(self.frame_width),
                    "height": int(self.frame_height),
                    "channels": 3,
                    "dtype": "uint8",
                    "format": "bgr8",
                    "stride_bytes": int(self.frame_width * self.frame_height * 3)
                }
                meta_path = os.path.join(self.output_dir, "RGB_Images", "raw_meta.json")
                try:
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)
                    self.rgb_meta_written = True
                except Exception as e:
                    print(f"警告: 写 raw_meta.json 失败: {e}")

            frame_index = self.rgb_frame_index
            self.rgb_frame_index += 1
            self.rgb_queue.put((frame_index, frame_bytes, rgb_ts, ts_aligned, int(seq)))

        except Exception as e:
            if self.running:
                print(f"错误: 处理RGB图像时出错: {e}")

    def slam_callback(self, msg):
        if self.recording_state != RecordingState.RECORDING:
            return
        try:
            timestamp = msg.poseMsg.header.stamp.to_sec()

            if self.slam_first_timestamp is None:
                self.slam_first_timestamp = timestamp
                self.try_calculate_vive_offset()

            self.calculate_slam_fps()

            x = msg.poseMsg.pose.position.x
            y = msg.poseMsg.pose.position.y
            z = msg.poseMsg.pose.position.z
            qx = msg.poseMsg.pose.orientation.x
            qy = msg.poseMsg.pose.orientation.y
            qz = msg.poseMsg.pose.orientation.z
            qw = msg.poseMsg.pose.orientation.w

            qpos_xv = [x, y, z, qx, qy, qz, qw]
            qpos_gripper = transform_slam_to_gripper(qpos_xv)
            current_pose = qpos_gripper

            if self.slam_prev_pose is not None and self.slam_prev_timestamp is not None:
                dt = timestamp - self.slam_prev_timestamp
                if dt > 0:
                    lin = self.calculate_linear_velocity(current_pose, self.slam_prev_pose, dt)
                    ang = self.calculate_angular_velocity(current_pose, self.slam_prev_pose, dt)
                    self.slam_velocity_history.append(lin)
                    self.slam_linear_velocity = self.smooth_velocity(self.slam_velocity_history)
                    self.slam_angular_velocity = ang
            self.slam_prev_pose = current_pose
            self.slam_prev_timestamp = timestamp

            x, y, z, qx, qy, qz, qw = qpos_gripper
            tum_line = f"{timestamp:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}\n"

            if self.slam_file_handle:
                self.slam_file_handle.write(tum_line)
            self.slam_count += 1
            self.slam_buffer.append((tum_line, timestamp))
        except Exception as e:
            if self.running:
                print(f"错误: 处理SLAM位姿时出错: {e}")

    def vive_callback(self, msg):
        if self.recording_state != RecordingState.RECORDING:
            return
        try:
            timestamp = msg.header.stamp.to_sec()
            if self.vive_first_timestamp is None:
                self.vive_first_timestamp = timestamp
                self.try_calculate_vive_offset()

            self.calculate_vive_fps()

            x = msg.pose.position.x
            y = msg.pose.position.y
            z = msg.pose.position.z
            qx = msg.pose.orientation.x
            qy = msg.pose.orientation.y
            qz = msg.pose.orientation.z
            qw = msg.pose.orientation.w

            qpos_vive = [x, y, z, qx, qy, qz, qw]
            qpos_gripper = transform_vive_to_gripper(qpos_vive)
            current_pose = qpos_gripper

            if self.vive_prev_pose is not None and self.vive_prev_timestamp is not None:
                dt = timestamp - self.vive_prev_timestamp
                if dt > 0:
                    lin = self.calculate_linear_velocity(current_pose, self.vive_prev_pose, dt)
                    ang = self.calculate_angular_velocity(current_pose, self.vive_prev_pose, dt)
                    self.vive_velocity_history.append(lin)
                    self.vive_linear_velocity = self.smooth_velocity(self.vive_velocity_history)
                    self.vive_angular_velocity = ang
            self.vive_prev_pose = current_pose
            self.vive_prev_timestamp = timestamp

            if not self.offset_ready:
                self.vive_pending_buffer.append((timestamp, qpos_gripper))
            else:
                aligned_timestamp = timestamp + self.vive_time_offset
                x, y, z, qx, qy, qz, qw = qpos_gripper
                tum_line = f"{aligned_timestamp:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}\n"
                if self.vive_file_handle:
                    self.vive_file_handle.write(tum_line)
                self.vive_count += 1
                self.vive_buffer.append((tum_line, aligned_timestamp))
        except Exception as e:
            if self.running:
                print(f"错误: 处理Vive位姿时出错: {e}")

    def clamp_callback(self, msg):
        if self.recording_state != RecordingState.RECORDING:
            return
        try:
            if isinstance(msg, rospy.AnyMsg):
                if self.clamp_msg_class is None:
                    type_str = msg._connection_header.get('type', '') if hasattr(msg, '_connection_header') else ''
                    if type_str:
                        self.clamp_msg_class = get_message_class(type_str)
                if self.clamp_msg_class is not None:
                    real_msg = self.clamp_msg_class()
                    real_msg.deserialize(msg._buff)
                else:
                    return
            else:
                real_msg = msg

            data_value = getattr(real_msg, 'data', None)
            if data_value is None:
                return

            ts_sec = None
            if hasattr(real_msg, 'header') and hasattr(real_msg.header, 'stamp'):
                if (real_msg.header.stamp.secs != 0) or (real_msg.header.stamp.nsecs != 0):
                    ts_sec = real_msg.header.stamp.to_sec()
            if ts_sec is None:
                ts_sec = rospy.get_time()

            tum_line = f"{ts_sec:.9f} {data_value}\n"
            if self.clamp_file_handle:
                self.clamp_file_handle.write(tum_line)
            self.clamp_count += 1
        except Exception as e:
            if self.running:
                print(f"错误: 处理Clamp数据时出错: {e}")

    def tof_callback(self, msg):
        if not self.enable_tof or self.recording_state != RecordingState.RECORDING:
            return
        try:
            ts = msg.header.stamp.to_sec()
            self.tof_queue.put((msg, ts))
            self.tof_count += 1
            self.calculate_tof_fps()
        except Exception as e:
            if self.running:
                print(f"错误: 处理ToF点云时出错: {e}")

    def calculate_rgb_fps(self, timestamp):
        now = time.time()
        self.rgb_timestamps.append(now)
        if now - self.rgb_last_fps_calc_time >= 1.0:
            if len(self.rgb_timestamps) >= 2:
                dt = now - self.rgb_timestamps[0]
                if dt > 0:
                    self.rgb_current_fps = (len(self.rgb_timestamps)-1) / dt
                    self.rgb_fps_history.append(self.rgb_current_fps)
                    if len(self.rgb_fps_history) > 0:
                        self.rgb_avg_fps = sum(self.rgb_fps_history) / len(self.rgb_fps_history)
            self.rgb_last_fps_calc_time = now

    def calculate_slam_fps(self):
        current_time = time.time()
        self.slam_timestamps.append(current_time)
        if current_time - self.slam_last_fps_calc_time >= 1.0:
            if len(self.slam_timestamps) >= 2:
                time_diff = current_time - self.slam_timestamps[0]
                if time_diff > 0:
                    self.slam_current_fps = (len(self.slam_timestamps) - 1) / time_diff
            self.slam_last_fps_calc_time = current_time

    def calculate_vive_fps(self):
        current_time = time.time()
        self.vive_timestamps.append(current_time)
        if current_time - self.vive_last_fps_calc_time >= 1.0:
            if len(self.vive_timestamps) >= 2:
                time_diff = current_time - self.vive_timestamps[0]
                if time_diff > 0:
                    self.vive_current_fps = (len(self.vive_timestamps) - 1) / time_diff
            self.vive_last_fps_calc_time = current_time

    def calculate_tof_fps(self):
        current_time = time.time()
        self.tof_timestamps.append(current_time)
        if current_time - self.tof_last_fps_calc_time >= 1.0:
            if len(self.tof_timestamps) >= 2:
                time_diff = current_time - self.tof_timestamps[0]
                if time_diff > 0:
                    self.tof_current_fps = (len(self.tof_timestamps) - 1) / time_diff
            self.tof_last_fps_calc_time = current_time

    def try_calculate_vive_offset(self):
        if self.slam_first_timestamp and self.vive_first_timestamp and not self.offset_ready:
            self.vive_time_offset = self.slam_first_timestamp - self.vive_first_timestamp
            self.offset_ready = True
            print(f"\n✓ Vive 时间戳对齐已启用")
            self.process_pending_vive_data()

    def save_vive_offset_info(self):
        """保存 Vive 时间戳对齐信息"""
        if not self.offset_ready:
            return
        try:
            vive_dir = os.path.join(self.output_dir, "Vive_Poses")
            offset_info_path = os.path.join(vive_dir, "offset_info.txt")
            with open(offset_info_path, 'w') as f:
                f.write("Vive 时间戳对齐信息\n")
                f.write("=" * 60 + "\n\n")
                f.write("对齐参数:\n")
                f.write(f"  SLAM 第一帧时间戳: {self.slam_first_timestamp:.6f}s\n")
                f.write(f"  Vive 第一帧时间戳: {self.vive_first_timestamp:.6f}s\n")
                f.write("说明:\n")
                f.write("  - 所有保存的 Vive 数据已使用对齐后的时间戳\n")
                f.write("  - 时间戳已与 SLAM 数据对齐\n")
                f.write("  - 坐标已转换到 Gripper 坐标系\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if self.device_label:
                    f.write(f"设备标签: {self.device_label}\n")
                f.write(f"XV 序列号: {self.xv_serial}\n")
                f.write(f"Vive 序列号: {self.vive_serial}\n")
            print(f"    ✓ 时间戳对齐信息已保存: {offset_info_path}")
        except Exception as e:
            print(f"    警告: 保存 offset 信息失败: {e}")

    def process_pending_vive_data(self):
        if not self.vive_pending_buffer:
            return
        processed = 0
        for timestamp, qpos in self.vive_pending_buffer:
            aligned_timestamp = timestamp + self.vive_time_offset
            x, y, z, qx, qy, qz, qw = qpos
            tum_line = f"{aligned_timestamp:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}\n"
            if self.vive_file_handle:
                self.vive_file_handle.write(tum_line)
            self.vive_count += 1
            self.vive_buffer.append((tum_line, aligned_timestamp))
            processed += 1
        self.vive_pending_buffer.clear()
        if self.device_label:
            print(f"[{self.device_label}] 已处理缓冲的 Vive 数据: {processed} 条")
        else:
            print(f"已处理缓冲的 Vive 数据: {processed} 条")

    def save_pointcloud_to_pcd(self, pointcloud_msg, filepath):
        try:
            pts = pc2.read_points(pointcloud_msg, field_names=("x","y","z"), skip_nans=False)
            arr = np.fromiter(pts, dtype=[('x','<f4'),('y','<f4'),('z','<f4')])
            width = arr.size
            # 统计
            if width > 0:
                valid = np.isfinite(arr.view('<f4').reshape(-1,3)).all(axis=1).sum()
                self.tof_stats['total_points'] += int(valid)
                z = arr['z'].astype(np.float32)
                finite = np.isfinite(z)
                if np.any(finite):
                    zf = z[finite]
                    self.tof_stats['min_depth'] = min(self.tof_stats['min_depth'], float(np.min(zf)))
                    self.tof_stats['max_depth'] = max(self.tof_stats['max_depth'], float(np.max(zf)))
                    self.tof_stats['total_invalid_points'] += int(width - zf.size)
            header = (
                "# .PCD v0.7 - Point Cloud Data file format\n"
                "VERSION 0.7\n"
                "FIELDS x y z\n"
                "SIZE 4 4 4\n"
                "TYPE F F F\n"
                "COUNT 1 1 1\n"
                f"WIDTH {width}\n"
                "HEIGHT 1\n"
                "VIEWPOINT 0 0 0 1 0 0 0\n"
                f"POINTS {width}\n"
                "DATA binary\n"
            ).encode("ascii")
            with open(filepath, "wb") as f:
                f.write(header)
                f.write(arr.tobytes())
        except Exception as e:
            print(f"错误: 保存点云文件失败 {filepath}: {e}")

    def save_all_data(self):
        print("正在保存数据...")
        try:
            # 等队列型写盘结束（RGB/ToF）
            if self.rgb_writer_thread:
                self.rgb_queue.join()
            if self.enable_tof and self.tof_writer_thread:
                self.tof_queue.join()

            # 强制把 Vive/SLAM/Clamp/Timestamps 的尾部刷到盘
            self.flush_all_stream_files()

            # 先关闭文件，保证磁盘上的文本完整可读
            self.cleanup_files()

            # 再从磁盘读取做融合
            self.save_merged_data()
            self.save_vive_offset_info()

            print("数据保存完成")
        except Exception as e:
            print(f"错误: 保存数据时出错: {e}")


    def encode_raw_to_mp4(self, framerate=60):
        """
        从 RAW 连续文件 frames.bgr8 合成 MP4（CFR=framerate）。
        优先 NVENC；若不可用或失败则回退 x264。
        """
        import os, json, shutil, subprocess

        try:
            rgb_dir = os.path.join(self.output_dir, "RGB_Images")
            raw_path = os.path.join(rgb_dir, "frames.bgr8")
            meta_path = os.path.join(rgb_dir, "raw_meta.json")
            if not (os.path.exists(raw_path) and os.path.exists(meta_path)):
                print("  警告: 缺少 RAW 或 meta，跳过视频合成")
                return

            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("format", "bgr8") != "bgr8":
                print("  警告: 当前仅支持 bgr8，跳过视频合成")
                return
            w = int(meta["width"]); h = int(meta["height"])

            ffmpeg_path = shutil.which("ffmpeg")
            if not ffmpeg_path:
                print("  警告: 未找到 ffmpeg，跳过视频合成")
                return

            mp4_path = os.path.join(rgb_dir, "video.mp4")

            def run_cmd(codec_args):
                cmd = [
                    ffmpeg_path, "-y", "-loglevel", "error",
                    "-f", "rawvideo",
                    "-pix_fmt", "bgr24",
                    "-s:v", f"{w}x{h}",
                    "-framerate", str(framerate),     # 输入帧率
                    "-i", raw_path,
                ] + codec_args + [
                    "-pix_fmt", "yuv420p",
                    "-r", str(framerate),             # 输出帧率
                    "-vsync", "cfr",
                    "-movflags", "+faststart",
                    "-f", "mp4",
                    mp4_path,
                ]
                subprocess.run(cmd, check=True)

            # 检测 NVENC 是否可用
            try:
                encoders_out = subprocess.run(
                    [ffmpeg_path, "-hide_banner", "-encoders"],
                    check=True, capture_output=True, text=True
                ).stdout
                has_nvenc = ("h264_nvenc" in encoders_out)
            except Exception:
                has_nvenc = False

            # NVENC
            if has_nvenc:
                nvenc_presets = ["llhq", "llhp", "hq", "hp", "fast", "medium", "default", "slow"]
                for preset in nvenc_presets:
                    try:
                        run_cmd([
                            "-c:v", "h264_nvenc",
                            "-preset", preset,
                            "-rc:v", "constqp", "-qp", "18",
                            # 如需码率控制可换："-rc:v","vbr","-cq","19"
                        ])
                        print(f"  ✓ NVENC 预设 {preset} 成功: {mp4_path}")
                        return
                    except subprocess.CalledProcessError as e:
                        continue
                    except Exception:
                        continue
                print("  NVENC 编码失败或驱动不可用，回退 libx264...")
            else:
                print("  未检测到 h264_nvenc 编码器，回退 libx264...")

            # CPU
            run_cmd(["-c:v", "libx264", "-crf", "18", "-preset", "veryfast"])
            print(f"  ✓ 已用 libx264 合成 MP4: {mp4_path}")

        except subprocess.CalledProcessError as e:
            print(f"  警告: 合成MP4失败。错误: {e}")
        except Exception as e:
            print(f"  警告: 合成过程中出错。错误: {e}")



    def save_merged_data(self):
        """从已写出的 TUM 文件进行流式双指针合并（内存O(1)）"""
        vive_dir = os.path.join(self.output_dir, "Vive_Poses")
        slam_dir = os.path.join(self.output_dir, "SLAM_Poses")
        merged_dir = os.path.join(self.output_dir, "Merged_Trajectory")
        os.makedirs(merged_dir, exist_ok=True)

        vive_path = os.path.join(vive_dir, "vive_data_tum.txt")
        slam_path = os.path.join(slam_dir, "slam_raw.txt")
        merged_path = os.path.join(merged_dir, "merged_trajectory.txt")

        if not (os.path.exists(vive_path) and os.path.exists(slam_path)):
            print("  警告: 无法生成融合轨迹（文件缺失）")
            return

        # 复位统计
        self.merge_stats.update({
            'total_merged': 0, 'averaged': 0, 'use_slam': 0,
            'use_vive': 0, 'both_high': 0, 'skipped': 0, 'no_match': 0
        })

        time_thresh = self.time_match_threshold

        def parse_line(line):
            parts = line.strip().split()
            if len(parts) < 8:
                return None, None
            ts = float(parts[0]); pose = [float(parts[i]) for i in range(1, 8)]
            return ts, pose

        with open(vive_path, 'r') as fv, open(slam_path, 'r') as fs, open(merged_path, 'w') as fm:
            from collections import deque as dq
            slam_win = dq(maxlen=3000)  # ~6s
            prev_slam_ts, prev_slam_pose = None, None
            prev_vive_ts, prev_vive_pose = None, None

            def slam_iter():
                for line in fs:
                    ts, pose = parse_line(line)
                    if ts is not None:
                        yield ts, pose

            s_iter = slam_iter()
            try:
                s_ts, s_pose = next(s_iter)
            except StopIteration:
                print("  警告: SLAM 文件为空")
                return

            for v_line in fv:
                v_ts, v_pose = parse_line(v_line)
                if v_ts is None:
                    continue

                # 推进 SLAM 到覆盖窗口
                while True:
                    slam_win.append((s_ts, s_pose))
                    try:
                        if s_ts >= v_ts + time_thresh:
                            break
                        s_ts, s_pose = next(s_iter)
                    except StopIteration:
                        break

                # 在窗口内找最近
                best = None; best_d = float('inf')
                for (ts, pose) in slam_win:
                    d = abs(ts - v_ts)
                    if d < best_d:
                        best, best_d = (ts, pose), d
                    if d < 0.002:
                        break

                if best is None or best_d > time_thresh:
                    self.merge_stats['no_match'] += 1
                    self.merge_stats['skipped'] += 1
                    prev_vive_ts, prev_vive_pose = v_ts, v_pose
                    continue

                s_ts_near, s_pose_near = best

                # 速度
                def lin_vel(curr_ts, curr_pose, p_ts, p_pose):
                    if p_ts is None or curr_ts <= p_ts:
                        return 0.0
                    return float(np.linalg.norm(np.array(curr_pose[:3]) - np.array(p_pose[:3])) / (curr_ts - p_ts))

                vive_vel = lin_vel(v_ts, v_pose, prev_vive_ts, prev_vive_pose)
                slam_vel = lin_vel(s_ts_near, s_pose_near, prev_slam_ts, prev_slam_pose)

                merged_pose, _, strategy = self.merge_poses(s_pose_near, v_pose, slam_vel, vive_vel)
                if merged_pose is not None:
                    x, y, z, qx, qy, qz, qw = merged_pose
                    fm.write(f"{v_ts:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
                    self.merge_stats['total_merged'] += 1
                else:
                    self.merge_stats['skipped'] += 1

                prev_vive_ts, prev_vive_pose = v_ts, v_pose
                prev_slam_ts, prev_slam_pose = s_ts_near, s_pose_near

            fm.flush()

        self.merged_count = self.merge_stats['total_merged']
        print(f"    Merged轨迹保存完成，共{self.merged_count}个位姿点")
        self.save_merge_stats()

    def save_merge_stats(self):
        try:
            merged_dir = os.path.join(self.output_dir, "Merged_Trajectory")
            stats_path = os.path.join(merged_dir, "merge_stats.txt")
            with open(stats_path, 'w') as f:
                f.write("轨迹融合统计信息（基于速度的动态融合 / 流式双指针）\n")
                f.write("=" * 60 + "\n\n")
                f.write("融合配置:\n")
                f.write(f"  速度阈值: {self.velocity_threshold} m/s\n")
                f.write(f"  时间匹配阈值: {self.time_match_threshold} s\n")
                f.write(f"  基准传感器: VIVE (100Hz)\n")
                f.write(f"  辅助传感器: SLAM (500Hz)\n\n")
                f.write("融合结果统计:\n")
                for k in ['total_merged','averaged','use_slam','use_vive','both_high','skipped','no_match']:
                    f.write(f"  {k}: {self.merge_stats[k]}\n")
                total = self.merge_stats['total_merged']
                if total > 0:
                    f.write("\n策略占比:\n")
                    f.write(f"  均值融合: {self.merge_stats['averaged']/total*100:.1f}%\n")
                    f.write(f"  使用SLAM: {self.merge_stats['use_slam']/total*100:.1f}%\n")
                    f.write(f"  使用VIVE: {self.merge_stats['use_vive']/total*100:.1f}%\n")
                f.write(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if self.device_label:
                    f.write(f"设备标签: {self.device_label}\n")
                f.write(f"XV 序列号: {self.xv_serial}\n")
                f.write(f"Vive 序列号: {self.vive_serial}\n")
            print(f"    ✓ 融合统计信息已保存: {stats_path}")
        except Exception as e:
            print(f"    警告: 保存融合统计信息失败: {e}")

    def display_status(self):
        print("\n" + "=" * 60)
        print("当前状态信息")
        print("=" * 60)
        print(f"录制状态: {self.recording_state.value}")
        if self.recording_state == RecordingState.RECORDING:
            dur = time.time() - self.recording_start_time if self.recording_start_time else 0.0
            print(f"录制时长: {dur:.1f}秒")
            print("当前数据计数（已落盘或已排队）:")
            print(f"  RGB图像: {self.rgb_count} 帧（队列 {self.rgb_queue.qsize()}/{self.rgb_queue.maxsize}）")
            print(f"  SLAM位姿: {self.slam_count} 条")
            print(f"  Vive位姿: {self.vive_count} 条")
            if self.enable_tof:
                print(f"  ToF点云: {self.tof_count} 个（队列 {self.tof_queue.qsize()}/{self.tof_queue.maxsize}）")
            else:
                print("  ToF点云: 已禁用")
            print(f"  Clamp数据: {self.clamp_count} 条")
        print("实时频率（估计）:")
        print(f"  RGB:  {self.rgb_current_fps:.1f} FPS (avg {self.rgb_avg_fps:.1f})")
        print(f"  SLAM: {self.slam_current_fps:.1f} Hz")
        print(f"  Vive: {self.vive_current_fps:.1f} Hz")
        if self.enable_tof:
            print(f"  ToF:  {self.tof_current_fps:.1f} Hz")
        mem = psutil.virtual_memory()
        print(f"系统内存使用: {mem.percent:.1f}%")
        print("=" * 60)

    def display_summary(self):
        """显示录制摘要"""
        if self.recording_start_time:
            recording_duration = self.recording_end_time - self.recording_start_time
        else:
            recording_duration = 0
        print(f"\n录制摘要:")
        print(f"  时长: {recording_duration:.1f}秒")
        print(f"  RGB图像: {self.rgb_count} 帧")
        print(f"  SLAM位姿: {self.slam_count} 个")
        print(f"  Vive位姿: {self.vive_count} 个")
        print(f"  Merged轨迹: {self.merged_count} 个")
        if self.enable_tof:
            print(f"  ToF点云: {self.tof_count} 个")
        else:
            print(f"  ToF点云: 已禁用")
        print(f"  Clamp数据: {self.clamp_count} 条")
        if self.rgb_count > 0:
            print(f"  RGB平均频率: {self.rgb_avg_fps:.1f} FPS")
        if self.merged_count > 0:
            print(f"\n  融合策略统计:")
            print(f"    均值融合: {self.merge_stats['averaged']} ({self.merge_stats['averaged']/self.merged_count*100:.1f}%)")
            print(f"    使用SLAM: {self.merge_stats['use_slam']} ({self.merge_stats['use_slam']/self.merged_count*100:.1f}%)")
            print(f"    使用VIVE: {self.merge_stats['use_vive']} ({self.merge_stats['use_vive']/self.merged_count*100:.1f}%)")
            print(f"    都高速(已跳过): {self.merge_stats['both_high']}")
            if self.merge_stats['skipped'] > 0:
                total_processed = self.merged_count + self.merge_stats['skipped']
                print(f"    总跳过率: {self.merge_stats['skipped']}/{total_processed} ({self.merge_stats['skipped']/total_processed*100:.1f}%)")
        if self.tof_count > 0 and self.tof_stats['total_points'] > 0:
            print(f"  ToF深度范围: {self.tof_stats['min_depth']:.3f}m - {self.tof_stats['max_depth']:.3f}m")
            print(f"  有效点数: {self.tof_stats['total_points']}")
        print(f"\n数据已保存到: {self.output_dir}")


class MultiDeviceRecorder:
    """双设备同步录制管理器"""

    def __init__(self, config, enable_tof=True, output_dir=None, encode_video=False):
        self.config = config
        self.enable_tof = enable_tof
        self.output_dir = output_dir
        self.encode_video = encode_video
        self.recorders = []
        self.running = True

        signal.signal(signal.SIGINT, self.signal_handler)

        print("=" * 60)
        print("摄像头 双设备同步数据采集器")
        print("=" * 60)

        self.create_recorders()

        print("=" * 60)
        print("所有设备已就绪")
        print("=" * 60)

    def create_recorders(self):
        devices = self.config['devices']

        if self.output_dir is not None:
            session_root = self.output_dir
            os.makedirs(session_root, exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_root = os.path.join(".", f"session_{timestamp}")
            os.makedirs(session_root, exist_ok=True)

        for device_key in sorted(devices.keys()):
            device_config = devices[device_key]
            label = device_config.get('label', device_key)
            xv_serial = device_config['xv_serial']

            # 每个设备各自的子目录
            device_output_dir = os.path.join(session_root, f"{label}_{xv_serial}")
            os.makedirs(device_output_dir, exist_ok=True)

            print(f"\n初始化设备 [{label}]:")
            print(f"  XV 序列号: {xv_serial}")
            print(f"  Vive 序列号: {device_config['vive_serial']}")
            print(f"  输出目录: {device_output_dir}")

            recorder = SingleDeviceRecorder(
                device_config=device_config,
                output_dir=device_output_dir,
                enable_tof=self.enable_tof,
                auto_start=True,
                encode_video=self.encode_video
            )
            self.recorders.append(recorder)

    def start_all_recording(self):
        print("\n" + "=" * 60)
        print("开始同步录制...")
        print("=" * 60)

        for r in self.recorders:
            r.prepare_recording()

        global_start_time = time.time()
        for r in self.recorders:
            r.start_recording(global_start_time)

        print(f"\n✓ {len(self.recorders)} 个设备已同时开始录制")
        print("按回车键停止录制...")
        print("=" * 60)

    def stop_all_recording(self):
        print("\n" + "=" * 60)
        print("停止所有设备录制...")
        print("=" * 60)

        tail_grace_s = 0.4
        time.sleep(tail_grace_s)

        for r in self.recorders:
            r.freeze_recording()
        for r in self.recorders:
            r.save_all_data()
            r.recording_state = RecordingState.FINISHED
            r.running = False

        print("\n✓ 所有设备录制完成")
        print("=" * 60)

    def display_summary(self):
        print("\n" + "=" * 60)
        print("录制摘要")
        print("=" * 60)
        for i, r in enumerate(self.recorders):
            print(f"\n设备 {i+1} [{r.device_label}]:")
            r.display_summary()

    def signal_handler(self, signum, frame):
        print("\n收到中断信号...")
        self.running = False

    def run(self):
        try:
            print("\n按回车键开始录制所有设备...")
            input()
            self.start_all_recording()
            input()
            self.stop_all_recording()
            self.display_summary()
        except rospy.ROSInterruptException:
            pass
        except KeyboardInterrupt:
            print("\n收到中断信号...")
        finally:
            self.cleanup()

    def cleanup(self):
        print("\n正在清理所有设备资源...")
        for r in self.recorders:
            r.cleanup()
        print("✓ 清理完成")


def load_config():
    config_paths = [
        "../start_process/config.json",
        "./start_process/config.json",
        "../config.json"
    ]
    config_file = None
    for path in config_paths:
        if os.path.exists(path):
            config_file = path
            break
    if config_file is None:
        print("错误: 找不到配置文件")
        print("\n请先运行以下命令之一：")
        print("  1. 启动完整系统（推荐）:")
        print("     cd start_process && ./unified_launcher.sh")
        print("  2. 仅运行设备配对:")
        print("     cd start_process && python3 device_pairing.py")
        sys.exit(1)
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        if 'single_device' not in config:
            print(f"错误: 配置文件缺少 'single_device' 字段: {config_file}")
            sys.exit(1)
        if 'devices' not in config:
            print(f"错误: 配置文件缺少 'devices' 字段: {config_file}")
            sys.exit(1)
        if 'device_0' not in config['devices']:
            print(f"错误: 配置文件缺少 'device_0': {config_file}")
            sys.exit(1)
        if not config['single_device'] and 'device_1' not in config['devices']:
            print(f"错误: 双设备模式但缺少 'device_1': {config_file}")
            sys.exit(1)
        print(f"✓ 成功加载配置: {config_file}")
        return config
    except json.JSONDecodeError as e:
        print(f"错误: 配置文件 JSON 格式错误: {e}")
        print(f"配置文件: {config_file}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 读取配置文件失败: {e}")
        print(f"配置文件: {config_file}")
        sys.exit(1)

def find_device_by_xv_serial(config, xv_serial):
    devices = config.get('devices', {})
    for _, device_config in devices.items():
        if device_config.get('xv_serial') == xv_serial:
            return device_config
    return None

def main():
    parser = argparse.ArgumentParser(description='摄像头数据采集器')
    parser.add_argument('--device', '-d', help='指定设备序列号（强制单设备模式）')
    parser.add_argument('--output', '-o', help='输出目录（单/双设备均可用，双设备时为会话根目录）')
    parser.add_argument('--tof', choices=['on', 'off'], default='off', help='是否采集ToF点云，默认off')
    parser.add_argument('--encode', choices=['on', 'off'], default='on', help='结束时是否从RAW合成MP4（默认on）')
    args = parser.parse_args()

    # 检查 ROS master
    try:
        rospy.get_master().getSystemState()
    except Exception:
        print("错误: ROS master未运行，请先启动ROS（roscore）")
        sys.exit(1)

    enable_tof = (args.tof == 'on')
    enable_encode = (args.encode == 'on')

    config = load_config()

    if args.device:
        print(f"单设备模式：{args.device}")
        device_config = find_device_by_xv_serial(config, args.device)
        if not device_config:
            print(f"警告: 配置中未找到设备 {args.device}，使用降级模式（无 Vive 配对信息）")
            device_config = {'xv_serial': args.device, 'vive_serial': 'UNKNOWN', 'label': ''}
        recorder = SingleDeviceRecorder(
            device_config=device_config,
            output_dir=args.output,
            enable_tof=enable_tof,
            auto_start=False,
            encode_video=enable_encode
        )
        recorder.run()
    else:
        if config['single_device']:
            print("单设备模式（根据配置文件）")
            device_config = config['devices']['device_0']
            recorder = SingleDeviceRecorder(
                device_config=device_config,
                output_dir=args.output,
                enable_tof=enable_tof,
                auto_start=False,
                encode_video=enable_encode
            )
            recorder.run()
        else:
            print("双设备同步模式（根据配置文件）")
            recorder = MultiDeviceRecorder(
                config=config,
                enable_tof=enable_tof,
                output_dir=args.output,
                encode_video=enable_encode
            )
            recorder.run()

    os._exit(0)


if __name__ == '__main__':
    main()
