#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设备信息获取工具
获取所有 XV 相机和 Vive Tracker 的序列号
"""

import sys
import time
import rospy
import openvr
from collections import defaultdict


def get_xvisio_serials():
    """获取所有 摄像头 相机的序列号"""
    try:
        # 获取所有话题
        topics = rospy.get_published_topics()
        
        # 过滤出 XV SDK 的话题，提取序列号
        xv_serials = set()
        for topic_name, _ in topics:
            if topic_name.startswith('/xv_sdk/'):
                parts = topic_name.split('/')
                if len(parts) >= 3:
                    serial = parts[2]
                    # 过滤掉 ROS 参数服务话题
                    if serial not in ['parameter_descriptions', 'parameter_updates', 'new_device']:
                        xv_serials.add(serial)
        
        return list(xv_serials)
    except Exception as e:
        print(f"❌ 获取 XV 序列号失败: {e}")
        return []


def get_vive_serials():
    """获取所有 Vive Tracker 的序列号（返回原始序列号和对应的索引）"""
    try:
        # 初始化 OpenVR
        openvr.init(openvr.VRApplication_Other)
        sys_obj = openvr.VRSystem()
        
        vive_trackers = {}
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if sys_obj.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_GenericTracker:
                try:
                    serial = sys_obj.getStringTrackedDeviceProperty(i, openvr.Prop_SerialNumber_String)
                    # 存储原始序列号
                    vive_trackers[serial] = i
                except Exception as e:
                    print(f"⚠️  无法获取设备 {i} 的序列号: {e}")
        
        openvr.shutdown()
        return vive_trackers
    except Exception as e:
        print(f"❌ 获取 Vive Tracker 失败: {e}")
        try:
            openvr.shutdown()
        except:
            pass
        return {}


def check_device_topics(xv_serials, vive_serials):
    """检查设备的 topic 是否正常发布"""
    print("\n" + "=" * 60)
    print("📡 检查设备 Topics 状态")
    print("=" * 60)
    
    # 检查 XV SLAM topics
    # 注意：XV SDK 发布的是 xv_sdk/PoseStampedConfidence 类型
    # 格式：{confidence: float, poseMsg: geometry_msgs/PoseStamped}
    print("\n【摄像头 相机 SLAM Topics】")
    for serial in xv_serials:
        topic = f"/xv_sdk/{serial}/slam/pose"
        try:
            msg = rospy.wait_for_message(topic, rospy.AnyMsg, timeout=2.0)
            print(f"  ✅ {serial}: {topic} (正常)")
        except:
            print(f"  ❌ {serial}: {topic} (无数据)")
    
    # 检查 Vive topics（topic 中使用下划线版本）
    print("\n【Vive Tracker Pose Topics】")
    for serial in vive_serials.keys():
        # 将连字符替换为下划线（与 vive_publisher.py 保持一致）
        topic_serial = serial.replace('-', '_')
        topic = f"/vive/{topic_serial}/pose"
        try:
            msg = rospy.wait_for_message(topic, rospy.AnyMsg, timeout=2.0)
            print(f"  ✅ {serial} -> {topic} (正常)")
        except:
            print(f"  ⚠️  {serial} -> {topic} (无数据 - 可能 vive_publisher.py 未运行)")


def main():
    print("\n" + "=" * 60)
    print("🔍 设备信息获取工具")
    print("=" * 60)
    
    # 初始化 ROS 节点（必须在 ROS 环境中运行）
    try:
        rospy.init_node('get_device_info', anonymous=True)
    except:
        print("\n❌ 错误: 无法初始化 ROS 节点")
        print("   请确保:")
        print("   1. roscore 已启动")
        print("   2. 在 ROS 环境中运行此脚本")
        return 1
    
    # 等待 ROS 准备就绪
    print("\n⏳ 等待 ROS 系统准备...")
    time.sleep(1.0)
    
    # 获取 摄像头 序列号
    print("\n📷 正在扫描 摄像头 相机...")
    xv_serials = get_xvisio_serials()
    
    if not xv_serials:
        print("  ❌ 未找到 摄像头 设备")
        print("     请确保:")
        print("     1. XV 相机已连接")
        print("     2. xv_sdk.launch 已启动")
    else:
        print(f"  ✅ 找到 {len(xv_serials)} 个 摄像头 设备:")
        for i, serial in enumerate(xv_serials, 1):
            print(f"     {i}. {serial}")
            print(f"        SLAM Topic: /xv_sdk/{serial}/slam/pose")
    
    # 获取 Vive Tracker 序列号
    print("\n🎮 正在扫描 Vive Trackers...")
    vive_trackers = get_vive_serials()
    
    if not vive_trackers:
        print("  ❌ 未找到 Vive Tracker")
        print("     请确保:")
        print("     1. 基站已开启")
        print("     2. Tracker 已配对")
        print("     3. SteamVR 正在运行")
    else:
        print(f"  ✅ 找到 {len(vive_trackers)} 个 Vive Tracker:")
        for i, (serial, idx) in enumerate(vive_trackers.items(), 1):
            topic_serial = serial.replace('-', '_')
            print(f"     {i}. {serial} (设备索引: {idx})")
            print(f"        Pose Topic: /vive/{topic_serial}/pose")
    
    # 检查 topics
    if xv_serials or vive_trackers:
        check_device_topics(xv_serials, vive_trackers)
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 设备统计")
    print("=" * 60)
    print(f"  摄像头 相机: {len(xv_serials)} 个")
    print(f"  Vive Tracker: {len(vive_trackers)} 个")
    
    # 判断配置场景
    print("\n" + "=" * 60)
    print("💡 配置建议")
    print("=" * 60)
    
    if len(xv_serials) == 1 and len(vive_trackers) == 1:
        print("  📌 单设备场景")
        print("     - 无需运行配对脚本")
        print("     - 可以直接开始数据采集")
        print(f"     - XV: {xv_serials[0]}")
        print(f"     - Vive: {list(vive_trackers.keys())[0]}")
    
    elif len(xv_serials) == 2 and len(vive_trackers) == 2:
        print("  📌 双设备场景")
        print("     - ⚠️ 需要运行配对脚本确定设备对应关系")
        print("     - 运行命令: python3 device_pairing.py")
        print("     - 配对完成后会保存到 config.json")
    
    elif len(xv_serials) != len(vive_trackers):
        print("  ⚠️  设备数量不匹配")
        print(f"     - XV 相机: {len(xv_serials)} 个")
        print(f"     - Vive Tracker: {len(vive_trackers)} 个")
        print("     - 请检查设备连接状态")
    
    else:
        print("  ⚠️  非标准配置（设备数量 > 2）")
        print("     - 当前脚本仅支持 1-2 个设备")
    
    print("\n" + "=" * 60)
    return 0


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

