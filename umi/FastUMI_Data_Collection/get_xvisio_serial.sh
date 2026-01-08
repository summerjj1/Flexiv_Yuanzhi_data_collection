#!/bin/bash

# 摄像头设备序列号获取脚本
# 作者: jiawei
# 日期: 2025-09-29

echo "正在获取摄像头设备序列号..."

# 检查ROS是否运行
if ! pgrep -x "rosmaster" > /dev/null; then
    echo "错误: ROS master未运行，请先启动ROS"
    exit 1
fi

# 获取所有摄像头设备序列号
echo "正在查询ROS话题..."
# 过滤掉ROS参数服务话题，只保留真正的设备序列号
SERIAL_NUMBERS=($(rostopic list 2>/dev/null | grep -o '/xv_sdk/[^/]*' | cut -d'/' -f3 | grep -v -E '^(parameter_descriptions|parameter_updates|new_device)$' | sort -u))

if [ ${#SERIAL_NUMBERS[@]} -eq 0 ]; then
    echo "错误: 未找到摄像头设备，请确保:"
    echo "  1. 摄像头摄像头已连接"
    echo "  2. xv_sdk.launch已启动"
    echo "  3. ROS环境正常"
    exit 1
fi

echo "找到 ${#SERIAL_NUMBERS[@]} 个摄像头设备:"
for i in "${!SERIAL_NUMBERS[@]}"; do
    echo "  设备 $((i+1)): ${SERIAL_NUMBERS[$i]}"
done

# 清空序列号文件
> xvisio_serial.txt

# 为每个设备显示详细信息
for i in "${!SERIAL_NUMBERS[@]}"; do
    SERIAL_NUMBER="${SERIAL_NUMBERS[$i]}"
    
    echo ""
    echo "=========================================="
    echo "设备 $((i+1)) 详细信息:"
    echo "  序列号: $SERIAL_NUMBER"
    echo "  命名空间: /xv_sdk/$SERIAL_NUMBER"
    
    # 显示可用的传感器
    echo ""
    echo "  可用传感器:"
    rostopic list 2>/dev/null | grep "/xv_sdk/$SERIAL_NUMBER" | grep -E "(camera|imu|slam|tof)" | sed 's|.*/||' | sort -u | while read sensor; do
        echo "    - $sensor"
    done
    
    # 显示主要话题
    echo ""
    echo "  主要数据话题:"
    echo "    彩色相机: /xv_sdk/$SERIAL_NUMBER/color_camera/image_color"
    echo "    鱼眼相机: /xv_sdk/$SERIAL_NUMBER/fisheye_cameras/left/image"
    echo "    RGBD相机: /xv_sdk/$SERIAL_NUMBER/rgbd_camera/image"
    echo "    ToF相机: /xv_sdk/$SERIAL_NUMBER/tof_camera/image"
    echo "    IMU传感器: /xv_sdk/$SERIAL_NUMBER/imu_sensor/data_raw"
    echo "    SLAM位姿: /xv_sdk/$SERIAL_NUMBER/slam/pose"
    
    # 保存序列号到文件
    echo "$SERIAL_NUMBER" >> xvisio_serial.txt
done

## 生成 config.json，保存所有序列号
CONFIG_FILE="config.json"
{
    echo "{"
    echo "  \"serial_numbers\": ["
    for i in "${!SERIAL_NUMBERS[@]}"; do
        SN="${SERIAL_NUMBERS[$i]}"
        if [ "$i" -lt $((${#SERIAL_NUMBERS[@]} - 1)) ]; then
            echo "    \"${SN}\",";
        else
            echo "    \"${SN}\"";
        fi
    done
    echo "  ]"
    echo "}"
} > "$CONFIG_FILE"

echo ""
echo "已生成配置文件: $CONFIG_FILE"
cat "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "所有序列号已保存到: xvisio_serial.txt"
echo "文件内容:"
cat xvisio_serial.txt | while read serial; do
    echo "  - $serial"
done
