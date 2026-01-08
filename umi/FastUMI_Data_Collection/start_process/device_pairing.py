#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设备配对脚本
通过运动检测自动识别 XV 相机和 Vive Tracker 的对应关系
"""

import sys
import time
import json
import argparse
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from collections import defaultdict
from pathlib import Path

# 尝试导入 XV SDK 的自定义消息类型
try:
    from xv_sdk.msg import PoseStampedConfidence
    XV_SDK_AVAILABLE = True
except ImportError:
    print("⚠️  警告: 无法导入 xv_sdk.msg.PoseStampedConfidence")
    print("   将使用 geometry_msgs/PoseStamped 作为后备")
    XV_SDK_AVAILABLE = False


class DevicePairing:
    def __init__(self, config_path="config.json"):
        self.config_path = Path(config_path)
        
        # 设备列表
        self.xv_serials = []
        self.vive_serials = []
        
        # 实时位置数据
        self.xv_poses = {}    # {serial: [最新的位置]}
        self.vive_poses = {}  # {serial: [最新的位置]}
        
        # 位移记录
        self.xv_displacements = defaultdict(float)
        self.vive_displacements = defaultdict(float)
        
        # 订阅者
        self.subscribers = []
        
    def load_devices(self):
        """从 ROS topics 和配置文件加载设备列表"""
        print("\n" + "=" * 60)
        print("🔍 正在扫描设备...")
        print("=" * 60)
        
        # 获取所有话题
        topics = rospy.get_published_topics()
        
        # 提取 XV 序列号
        for topic_name, _ in topics:
            if '/xv_sdk/' in topic_name and '/slam/pose' in topic_name:
                parts = topic_name.split('/')
                if len(parts) >= 3:
                    serial = parts[2]
                    if serial not in self.xv_serials:
                        self.xv_serials.append(serial)
        
        # 提取 Vive 序列号
        for topic_name, _ in topics:
            if '/vive/' in topic_name and '/pose' in topic_name:
                parts = topic_name.split('/')
                if len(parts) >= 3:
                    serial = parts[2]
                    # 过滤掉可能的 rpy topic
                    if serial not in self.vive_serials and topic_name.endswith('/pose'):
                        self.vive_serials.append(serial)
        
        print(f"\n📷 找到 {len(self.xv_serials)} 个 摄像头 设备:")
        for i, serial in enumerate(self.xv_serials, 1):
            print(f"   {i}. {serial}")
        
        print(f"\n🎮 找到 {len(self.vive_serials)} 个 Vive Tracker:")
        for i, serial in enumerate(self.vive_serials, 1):
            print(f"   {i}. {serial}")
        
        # 验证设备数量
        if len(self.xv_serials) != len(self.vive_serials):
            print(f"\n❌ 错误: XV 设备数量 ({len(self.xv_serials)}) 与 Vive 设备数量 ({len(self.vive_serials)}) 不匹配")
            return False
        
        if len(self.xv_serials) == 0:
            print("\n❌ 错误: 未找到任何设备")
            return False
        
        if len(self.xv_serials) == 1:
            print("\n📌 检测到单设备场景，无需配对")
            self._save_single_device_config()
            return False
        
        if len(self.xv_serials) > 2:
            print(f"\n⚠️  警告: 检测到 {len(self.xv_serials)} 个设备，当前仅支持 1-2 个设备")
        
        return True
    
    def _save_single_device_config(self):
        """保存单设备配置"""
        config = {
            "single_device": True,
            "devices": {
                "device_0": {
                    "label": "main",
                    "xv_serial": self.xv_serials[0],
                    "vive_serial": self.vive_serials[0]
                }
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 单设备配置已保存到 {self.config_path}")
        print(f"   XV: {self.xv_serials[0]}")
        print(f"   Vive: {self.vive_serials[0]}")
    
    def subscribe_topics(self):
        """订阅所有设备的 pose topics"""
        print("\n" + "=" * 60)
        print("📡 订阅设备 Topics...")
        print("=" * 60)
        
        # 订阅 XV SLAM poses（使用 XV SDK 的自定义消息类型）
        for serial in self.xv_serials:
            topic = f"/xv_sdk/{serial}/slam/pose"
            if XV_SDK_AVAILABLE:
                sub = rospy.Subscriber(
                    topic, 
                    PoseStampedConfidence, 
                    lambda msg, s=serial: self._xv_pose_callback(msg, s)
                )
            else:
                # 后备方案：使用标准 PoseStamped
                sub = rospy.Subscriber(
                    topic, 
                    PoseStamped, 
                    lambda msg, s=serial: self._xv_pose_callback(msg, s)
                )
            self.subscribers.append(sub)
            print(f"  📷 订阅: {topic}")
        
        # 订阅 Vive poses（topic 中使用下划线版本）
        for serial in self.vive_serials:
            # serial 是从 topic 提取的，已经是下划线版本
            topic = f"/vive/{serial}/pose"
            sub = rospy.Subscriber(
                topic,
                PoseStamped,
                lambda msg, s=serial: self._vive_pose_callback(msg, s)
            )
            self.subscribers.append(sub)
            print(f"  🎮 订阅: {topic}")
        
        # 等待接收数据
        print("\n⏳ 等待接收数据...")
        wait_time = 3.0
        for i in range(int(wait_time * 10)):
            time.sleep(0.1)
            if len(self.xv_poses) == len(self.xv_serials) and \
               len(self.vive_poses) == len(self.vive_serials):
                print("  ✅ 所有设备数据接收正常")
                return True
        
        # 检查缺失的设备
        missing = []
        for serial in self.xv_serials:
            if serial not in self.xv_poses:
                missing.append(f"XV: {serial}")
        for serial in self.vive_serials:
            if serial not in self.vive_poses:
                missing.append(f"Vive: {serial}")
        
        if missing:
            print("\n❌ 以下设备未接收到数据:")
            for m in missing:
                print(f"   - {m}")
            return False
        
        return True
    
    def _xv_pose_callback(self, msg, serial):
        """XV SLAM pose 回调"""
        # PoseStampedConfidence 格式: {confidence: float, poseMsg: PoseStamped}
        if hasattr(msg, 'poseMsg'):
            # xv_sdk/PoseStampedConfidence 格式
            pos = msg.poseMsg.pose.position
        elif hasattr(msg, 'pose'):
            # 标准 geometry_msgs/PoseStamped 格式
            pos = msg.pose.position
        else:
            rospy.logwarn(f"未知的消息格式: {type(msg)}")
            return
        self.xv_poses[serial] = np.array([pos.x, pos.y, pos.z])
    
    def _vive_pose_callback(self, msg, serial):
        """Vive pose 回调"""
        pos = msg.pose.position
        self.vive_poses[serial] = np.array([pos.x, pos.y, pos.z])
    
    def record_baseline(self, duration=0.5):
        """记录静置状态的基准位置"""
        print("\n" + "=" * 60)
        print("📍 记录基准位置")
        print("=" * 60)
        print(f"⏱️  正在记录 {duration} 秒...")
        print("   请保持所有设备静止！")
        
        # 记录初始位置
        baseline_xv = {s: [] for s in self.xv_serials}
        baseline_vive = {s: [] for s in self.vive_serials}
        
        start_time = time.time()
        rate = rospy.Rate(50)  # 50Hz
        
        while time.time() - start_time < duration and not rospy.is_shutdown():
            for serial in self.xv_serials:
                if serial in self.xv_poses:
                    baseline_xv[serial].append(self.xv_poses[serial].copy())
            
            for serial in self.vive_serials:
                if serial in self.vive_poses:
                    baseline_vive[serial].append(self.vive_poses[serial].copy())
            
            rate.sleep()
        
        # 计算平均位置作为基准
        self.baseline_xv = {}
        self.baseline_vive = {}
        
        for serial in self.xv_serials:
            if baseline_xv[serial]:
                self.baseline_xv[serial] = np.mean(baseline_xv[serial], axis=0)
                print(f"  📷 XV {serial}: {self.baseline_xv[serial]}")
        
        for serial in self.vive_serials:
            if baseline_vive[serial]:
                self.baseline_vive[serial] = np.mean(baseline_vive[serial], axis=0)
                print(f"  🎮 Vive {serial}: {self.baseline_vive[serial]}")
        
        print("  ✅ 基准位置记录完成")
    
    def detect_motion(self, duration=5.0, target_device="device_0"):
        """检测运动并记录位移"""
        print("\n" + "=" * 60)
        print("🚀 运动检测")
        print("=" * 60)
        print(f"⏱️  检测时长: {duration} 秒")
        print(f"📍 请移动【{target_device}（左侧）】设备，幅度约 20-30cm")
        print("   另一个设备请保持静止！")
        print("\n倒计时: ", end="", flush=True)
        
        # 倒计时
        for i in range(3, 0, -1):
            print(f"{i}... ", end="", flush=True)
            time.sleep(1.0)
        print("开始！\n")
        
        # 重置位移记录
        self.xv_displacements = defaultdict(float)
        self.vive_displacements = defaultdict(float)
        
        # 上一帧位置
        prev_xv = {s: self.baseline_xv.get(s, np.zeros(3)) for s in self.xv_serials}
        prev_vive = {s: self.baseline_vive.get(s, np.zeros(3)) for s in self.vive_serials}
        
        start_time = time.time()
        rate = rospy.Rate(100)  # 100Hz
        
        while time.time() - start_time < duration and not rospy.is_shutdown():
            elapsed = time.time() - start_time
            
            # 计算 XV 位移
            for serial in self.xv_serials:
                if serial in self.xv_poses:
                    curr_pos = self.xv_poses[serial]
                    displacement = np.linalg.norm(curr_pos - prev_xv[serial])
                    self.xv_displacements[serial] += displacement
                    prev_xv[serial] = curr_pos.copy()
            
            # 计算 Vive 位移
            for serial in self.vive_serials:
                if serial in self.vive_poses:
                    curr_pos = self.vive_poses[serial]
                    displacement = np.linalg.norm(curr_pos - prev_vive[serial])
                    self.vive_displacements[serial] += displacement
                    prev_vive[serial] = curr_pos.copy()
            
            # 显示进度
            progress = int((elapsed / duration) * 40)
            bar = "█" * progress + "░" * (40 - progress)
            print(f"\r  [{bar}] {elapsed:.1f}s / {duration:.1f}s", end="", flush=True)
            
            rate.sleep()
        
        print("\n\n✅ 运动检测完成")
        
        # 显示结果
        print("\n📊 位移统计:")
        print("\n  摄像头 设备:")
        for serial in self.xv_serials:
            disp = self.xv_displacements[serial]
            status = "🟢 运动" if disp > 0.15 else "🔴 静止"
            print(f"    {status} {serial}: {disp:.3f}m")
        
        print("\n  Vive Tracker:")
        for serial in self.vive_serials:
            disp = self.vive_displacements[serial]
            status = "🟢 运动" if disp > 0.15 else "🔴 静止"
            print(f"    {status} {serial}: {disp:.3f}m")
    
    def analyze_pairing(self, motion_threshold=0.15, static_threshold=0.08):
        """分析配对关系"""
        print("\n" + "=" * 60)
        print("🔬 分析配对关系")
        print("=" * 60)
        
        # 找出运动的设备
        moving_xv = [s for s in self.xv_serials if self.xv_displacements[s] > motion_threshold]
        moving_vive = [s for s in self.vive_serials if self.vive_displacements[s] > motion_threshold]
        
        # 找出静止的设备
        static_xv = [s for s in self.xv_serials if self.xv_displacements[s] < static_threshold]
        static_vive = [s for s in self.vive_serials if self.vive_displacements[s] < static_threshold]
        
        print(f"\n运动阈值: {motion_threshold}m")
        print(f"静止阈值: {static_threshold}m")
        
        print(f"\n运动设备:")
        print(f"  XV: {moving_xv}")
        print(f"  Vive: {moving_vive}")
        
        print(f"\n静止设备:")
        print(f"  XV: {static_xv}")
        print(f"  Vive: {static_vive}")
        
        # 验证配对结果
        if len(moving_xv) != 1 or len(moving_vive) != 1:
            print("\n❌ 配对失败: 运动设备数量不正确")
            print("   请确保只移动一个设备，另一个保持静止")
            return None
        
        if len(static_xv) != 1 or len(static_vive) != 1:
            print("\n❌ 配对失败: 静止设备数量不正确")
            return None
        
        # 配对结果
        pairing = {
            "device_0": {
                "label": "left_hand",
                "xv_serial": moving_xv[0],
                "vive_serial": moving_vive[0]
            },
            "device_1": {
                "label": "right_hand",
                "xv_serial": static_xv[0],
                "vive_serial": static_vive[0]
            }
        }
        
        print("\n" + "=" * 60)
        print("✅ 配对成功!")
        print("=" * 60)
        print("\nDevice_0 (左侧 / 运动的):")
        print(f"  XV: {pairing['device_0']['xv_serial']}")
        print(f"  Vive: {pairing['device_0']['vive_serial']}")
        print(f"  (Topic 中为: {pairing['device_0']['vive_serial']})")
        print("\nDevice_1 (右侧 / 静止的):")
        print(f"  XV: {pairing['device_1']['xv_serial']}")
        print(f"  Vive: {pairing['device_1']['vive_serial']}")
        print(f"  (Topic 中为: {pairing['device_1']['vive_serial']})")
        
        return pairing
    
    def save_config(self, pairing):
        """保存配对配置到文件"""
        config = {
            "single_device": False,
            "devices": pairing
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 配置已保存到: {self.config_path}")
    
    def run(self):
        """运行完整的配对流程"""
        print("\n" + "=" * 60)
        print("🎯 设备自动配对工具")
        print("=" * 60)
        print("\n此工具将通过运动检测自动识别 XV 和 Vive 的对应关系")
        print("适用于双设备场景")
        
        # 加载设备
        if not self.load_devices():
            return 0
        
        # 订阅 topics
        if not self.subscribe_topics():
            return 1
        
        # 用户确认
        print("\n" + "=" * 60)
        input("📌 准备开始配对流程，请按 Enter 继续...")
        
        # 记录基准
        self.record_baseline(duration=0.5)
        
        print("\n" + "=" * 60)
        input("📌 准备开始运动检测，请按 Enter 继续...")
        
        # 运动检测
        self.detect_motion(duration=5.0)
        
        # 分析配对
        pairing = self.analyze_pairing()
        
        if pairing is None:
            print("\n❌ 配对失败，请重新运行脚本")
            return 1
        
        # 确认保存
        print("\n" + "=" * 60)
        response = input("📌 是否保存此配对结果? (y/n): ").strip().lower()
        
        if response == 'y' or response == 'yes':
            self.save_config(pairing)
            print("\n✅ 配对完成！可以开始数据采集")
            return 0
        else:
            print("\n⚠️  配对结果未保存，请重新运行")
            return 1


def main():
    parser = argparse.ArgumentParser(description="设备自动配对工具")
    parser.add_argument("--config", type=str, default="config.json", 
                       help="配置文件路径（默认: config.json）")
    args = parser.parse_args()
    
    # 初始化 ROS
    try:
        rospy.init_node('device_pairing', anonymous=True)
    except:
        print("\n❌ 错误: 无法初始化 ROS 节点")
        print("   请确保 roscore 已启动")
        return 1
    
    # 运行配对
    pairing_tool = DevicePairing(config_path=args.config)
    return pairing_tool.run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

