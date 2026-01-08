#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import sys
import os
import threading

# --- 1. 路径设置 ---
PROJECT_PATH = "/home/fx/Fast_umi/FastUMI_Data_Collection/data_collector_opt"
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

try:
    from pose_merge import transform_slam_to_gripper
    # 导入处理动态消息所需的库
    from roslib.message import get_message_class
except ImportError as e:
    print(f"错误: 无法导入依赖模块。请确认路径 {PROJECT_PATH} 正确。")
    sys.exit(1)

from xv_sdk.msg import PoseStampedConfidence
# 使用 AnyMsg 来接收 Clamp 数据，这样不管它是什么类型（Bool/Float/String）都能收
from rospy.msg import AnyMsg

# --- 全局变量 ---
# 用于在两个回调函数之间共享数据
latest_clamp_data = "No Data" 
clamp_msg_class = None # 用于缓存动态加载的消息类

def clamp_callback(msg):
    """
    完全复用 SingleDeviceRecorder 中的 Clamp 处理逻辑
    """
    global latest_clamp_data, clamp_msg_class
    
    try:
        # 动态解析 AnyMsg
        if isinstance(msg, AnyMsg):
            if clamp_msg_class is None:
                # 从连接头中获取消息类型字符串 (例如 'std_msgs/Float32')
                type_str = msg._connection_header.get('type', '')
                if type_str:
                    clamp_msg_class = get_message_class(type_str)
            
            if clamp_msg_class is not None:
                real_msg = clamp_msg_class()
                real_msg.deserialize(msg._buff)
            else:
                return # 还没获取到类型，跳过
        else:
            real_msg = msg

        # 尝试提取 data 字段
        if hasattr(real_msg, 'data'):
            # 这里格式化一下，保留两位小数(如果是浮点数)或者直接转字符串
            val = real_msg.data
            if isinstance(val, float):
                latest_clamp_data = f"{val:.4f}"
            else:
                latest_clamp_data = str(val)
                
    except Exception as e:
        latest_clamp_data = f"Err: {str(e)[:10]}"

def slam_callback(msg):
    """
    处理 SLAM 数据并负责刷新屏幕显示
    """
    global latest_clamp_data
    
    try:
        # 1. 提取原始 SLAM 数据
        x = msg.poseMsg.pose.position.x
        y = msg.poseMsg.pose.position.y
        z = msg.poseMsg.pose.position.z
        qx = msg.poseMsg.pose.orientation.x
        qy = msg.poseMsg.pose.orientation.y
        qz = msg.poseMsg.pose.orientation.z
        qw = msg.poseMsg.pose.orientation.w

        # 2. 坐标转换 (相机 -> 夹爪)
        qpos_xv = [x, y, z, qx, qy, qz, qw]
        qpos_gripper = transform_slam_to_gripper(qpos_xv)
        gx, gy, gz, gqx, gqy, gqz, gqw = qpos_gripper

        # 3. 实时打印 (包含 Clamp 数据)
        # 格式说明：
        # [Pos]: 夹爪位置 XYZ
        # [Rot]: 夹爪姿态四元数
        # [Clamp]: 夹爪开合数值
        info_str = (
            f"\r"
            f"Pos: [{gx: .4f}, {gy: .4f}, {gz: .4f}] | "
            f"Rot: [{gqx:.2f}, {gqy:.2f}, {gqz:.2f}, {gqw:.2f}] | "
            f"Clamp: [{latest_clamp_data}]"
            f"      " # 尾部空格清除旧字符
        )
        sys.stdout.write(info_str)
        sys.stdout.flush()

    except Exception as e:
        pass

if __name__ == '__main__':
    rospy.init_node('gripper_info_printer', anonymous=True)

    # --- 配置 ---
    XV_SERIAL = "250801DR48FP25002239"
    SLAM_TOPIC = f"/xv_sdk/{XV_SERIAL}/slam/pose"
    CLAMP_TOPIC = f"/xv_sdk/{XV_SERIAL}/clamp/Data"

    print("="*80)
    print(f"SLAM  话题: {SLAM_TOPIC}")
    print(f"Clamp 话题: {CLAMP_TOPIC}")
    print("-" * 80)
    print("正在输出夹爪坐标系位姿与开合状态...")
    print("按 Ctrl+C 退出")
    print("="*80)

    # 订阅 Clamp (使用 AnyMsg 以兼容所有类型)
    rospy.Subscriber(CLAMP_TOPIC, AnyMsg, clamp_callback)
    
    # 订阅 SLAM
    rospy.Subscriber(SLAM_TOPIC, PoseStampedConfidence, slam_callback)

    rospy.spin()
    print("\n退出。")