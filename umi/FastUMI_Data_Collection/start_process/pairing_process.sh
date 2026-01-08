#!/bin/bash

# 设备配对系统快速启动脚本
# 自动引导用户完成设备配对流程

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         设备配对系统 - 快速启动向导                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 检查 ROS 环境
if ! pgrep -x "rosmaster" > /dev/null; then
    echo "❌ 错误: ROS master 未运行"
    echo "   请先在另一个终端运行: roscore"
    exit 1
fi

echo "✅ ROS master 正在运行"

# 检查 Python 脚本
if [ ! -f "get_device_info.py" ]; then
    echo "❌ 错误: 找不到 get_device_info.py"
    exit 1
fi

if [ ! -f "device_pairing.py" ]; then
    echo "❌ 错误: 找不到 device_pairing.py"
    exit 1
fi

if [ ! -f "vive_publisher.py" ]; then
    echo "❌ 错误: 找不到 vive_publisher.py"
    exit 1
fi

echo "✅ 所有脚本文件存在"
echo ""

# 步骤 1: 扫描设备
echo "════════════════════════════════════════════════════════════"
echo "📍 步骤 1/3: 扫描设备"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "正在扫描连接的设备..."
echo ""

python3 get_device_info.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 设备扫描失败，请检查："
    echo "   1. XV 相机是否已连接并启动 xv_sdk.launch"
    echo "   2. Vive Tracker 是否在 SteamVR 中可见"
    echo "   3. vive_publisher.py 是否正在运行"
    exit 1
fi

echo ""
read -p "📌 按 Enter 继续..."

# 检查是否需要配对
if [ -f "config.json" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "⚠️  发现已存在的配置文件"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "当前配置:"
    cat config.json
    echo ""
    read -p "是否重新配对? (y/n): " RECONFIG
    
    if [ "$RECONFIG" != "y" ] && [ "$RECONFIG" != "Y" ]; then
        echo ""
        echo "✅ 使用现有配置"
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "💡 下一步"
        echo "════════════════════════════════════════════════════════════"
        echo ""
        echo "配置已完成，可以开始数据采集："
        echo ""
        echo "  cd ../data_collector"
        echo "  python3 single_session_data_collector.py"
        echo ""
        exit 0
    fi
fi

# 步骤 2: 运行配对
echo ""
echo "════════════════════════════════════════════════════════════"
echo "📍 步骤 2/3: 设备配对"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "即将启动自动配对流程..."
echo ""
echo "📋 准备工作："
echo "   1. 将两个设备放在稳定表面"
echo "   2. 确定哪个是【左侧】设备（将被设置为 device_0）"
echo "   3. 准备移动左侧设备（幅度 20-30cm）"
echo ""
read -p "准备好了吗？按 Enter 开始配对..."

python3 device_pairing.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 配对失败"
    echo ""
    echo "💡 故障排查："
    echo "   1. 确保只移动一个设备"
    echo "   2. 移动幅度要足够（20-30cm）"
    echo "   3. 另一个设备完全保持静止"
    echo "   4. 检查基站视野是否良好"
    echo ""
    echo "可以重新运行此脚本重试: ./quick_start.sh"
    exit 1
fi

# 步骤 3: 完成
echo ""
echo "════════════════════════════════════════════════════════════"
echo "📍 步骤 3/3: 完成"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "🎉 配对完成！"
echo ""
echo "当前配置:"
cat config.json
echo ""

echo "════════════════════════════════════════════════════════════"
echo "💡 下一步"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "配置已完成，可以开始数据采集："
echo ""
echo "  cd ../data_collector"
echo "  python3 single_session_data_collector.py"
echo ""
echo "如需重新配对，再次运行: ./quick_start.sh"
echo ""

