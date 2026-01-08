#!/bin/bash

##############################################################################
# 统一启动脚本 - 摄像头摄像头 + Vive发布器
# 功能：
# 1. 按顺序启动摄像头摄像头和Vive发布器
# 2. 自动检测ROS环境和catkin_ws
# 3. 监控服务状态并提供错误处理
# 4. 支持进程管理和优雅退出
##############################################################################

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/launcher.log"
PID_FILE="$SCRIPT_DIR/launcher.pid"
xvisio_PID_FILE="$SCRIPT_DIR/xvisio.pid"
VIVE_PID_FILE="$SCRIPT_DIR/vive.pid"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印彩色消息
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 退出标志
EXIT_REQUESTED=false

##############################################################################
# 帮助信息
##############################################################################
show_help() {
    echo "统一启动脚本 - 摄像头摄像头 + Vive发布器"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help           显示此帮助信息"
    echo "  --no-gui             禁用图形界面终端（后台运行）"
    echo ""
    echo "示例:"
    echo "  $0                    # 启动所有服务（GUI模式）"
    echo "  $0 --no-gui           # 启动所有服务（后台模式）"
    echo ""
    echo "说明:"
    echo "  - 默认在新终端窗口中启动服务"
    echo "  - 使用 --no-gui 在后台运行"
    echo "  - 按 Ctrl+C 停止所有服务"
}

##############################################################################
# 智能查找catkin_ws目录（复用现有逻辑）
##############################################################################
find_catkin_ws() {
    # 首先检查当前目录
    if [ -d "catkin_ws" ]; then
        echo "catkin_ws"
        return 0
    fi
    
    # 检查上级目录
    if [ -d "../catkin_ws" ]; then
        echo "../catkin_ws"
        return 0
    fi
    
    # 检查用户主目录
    if [ -d "$HOME/catkin_ws" ]; then
        echo "$HOME/catkin_ws"
        return 0
    fi
    
    # 使用find命令在用户目录下搜索
    local found_path=$(find "$HOME" -maxdepth 3 -name "catkin_ws" -type d 2>/dev/null | head -1)
    if [ -n "$found_path" ]; then
        echo "$found_path"
        return 0
    fi
    
    return 1
}

##############################################################################
# 检查ROS环境
##############################################################################
check_ros_environment() {
    # 检查ROS是否安装
    if ! command -v rospack >/dev/null 2>&1; then
        print_error "ROS未安装或未正确配置"
        return 1
    fi
    
    # 检查ROS环境变量
    if [ -z "$ROS_DISTRO" ]; then
        print_warning "ROS_DISTRO环境变量未设置，尝试source ROS环境..."
        if [ -f "/opt/ros/noetic/setup.bash" ]; then
            source /opt/ros/noetic/setup.bash
            print_info "已加载ROS Noetic环境"
        else
            print_error "找不到ROS环境文件"
            return 1
        fi
    fi
    
    return 0
}

##############################################################################
# 检查catkin_ws
##############################################################################
check_catkin_ws() {
    print_info "正在查找catkin_ws目录..."
    CATKIN_WS_PATH=$(find_catkin_ws)
    
    if [ $? -ne 0 ] || [ -z "$CATKIN_WS_PATH" ]; then
        print_error "找不到catkin_ws目录"
        print_error "请确保以下位置之一存在catkin_ws目录:"
        print_error "  - 当前目录: $(pwd)/catkin_ws"
        print_error "  - 上级目录: $(dirname $(pwd))/catkin_ws"
        print_error "  - 用户主目录: $HOME/catkin_ws"
        return 1
    fi
    
    print_success "找到catkin_ws目录: $CATKIN_WS_PATH"
    
    # 检查devel/setup.sh是否存在
    if [ ! -f "$CATKIN_WS_PATH/devel/setup.sh" ]; then
        print_error "找不到devel/setup.sh文件，请确保ROS工作空间已正确编译"
        return 1
    fi
    
    return 0
}

##############################################################################
# 检查ROS master状态
##############################################################################
check_rosmaster() {
    if pgrep -x "rosmaster" > /dev/null; then
        print_success "ROS master正在运行"
        return 0
    else
        print_warning "ROS master未运行"
        return 1
    fi
}

##############################################################################
# 等待ROS master启动
##############################################################################
wait_for_rosmaster() {
    local timeout=30
    local count=0
    
    print_info "等待ROS master启动..."
    
    while [ $count -lt $timeout ]; do
        if check_rosmaster; then
            print_success "ROS master已启动，等待6秒确保服务就绪..."
            sleep 6
            return 0
        fi
        
        sleep 1
        count=$((count + 1))
        printf "\r等待ROS master... ($count/$timeout)"
    done
    
    echo ""
    print_error "等待ROS master超时"
    return 1
}

##############################################################################
# 启动摄像头摄像头
##############################################################################
start_xvisio_camera() {
    print_info "启动摄像头摄像头..."
    
    # 检查是否已有 xv_sdk 进程在运行
    if pgrep -f "/xv_sdk/xv_sdk" > /dev/null 2>&1; then
        print_warning "检测到已有 xv_sdk 进程在运行"
        local existing_count=$(pgrep -f "/xv_sdk/xv_sdk" | wc -l)
        print_warning "现有进程数: $existing_count"
        print_info "尝试清理现有进程..."
        
        # 清理现有进程
        pkill -f "/xv_sdk/xv_sdk" 2>/dev/null || true
        sleep 3
        pkill -9 -f "/xv_sdk/xv_sdk" 2>/dev/null || true
        sleep 1
        
        # 清理 ROS 节点
        if rosnode list 2>/dev/null | grep -q "xv_sdk"; then
            print_info "清理旧的 xv_sdk ROS 节点..."
            rosnode kill /xv_sdk 2>/dev/null || true
            sleep 1
        fi
        
        print_success "旧进程已清理"
    fi
    
    # 切换到catkin_ws目录
    cd "$CATKIN_WS_PATH"
    
    # 启动摄像头 SDK
    if [ "$NO_GUI" = true ]; then
        # 非GUI模式：在后台启动
        source devel/setup.sh
        nohup roslaunch xv_sdk xv_sdk.launch > "$LOG_FILE" 2>&1 &
        local xvisio_pid=$!
        echo $xvisio_pid > "$xvisio_PID_FILE"
        print_info "摄像头摄像头已在后台启动，PID: $xvisio_pid"
    else
        # GUI模式：在新终端窗口启动
        gnome-terminal --title="摄像头摄像头" -- bash -c "
            echo '正在启动摄像头传感器模组...';
            cd '$CATKIN_WS_PATH';
            source devel/setup.sh;
            echo 'ROS环境已加载，启动摄像头传感器模组...';
            roslaunch xv_sdk xv_sdk.launch;
            echo '按任意键关闭此窗口...';
            read -n 1
        " 2>/dev/null || \
        xterm -title "摄像头摄像头" -e "
            echo '正在启动摄像头传感器模组...';
            cd '$CATKIN_WS_PATH';
            source devel/setup.sh;
            echo 'ROS环境已加载，启动摄像头传感器模组...';
            roslaunch xv_sdk xv_sdk.launch;
            echo '按任意键关闭此窗口...';
            read -n 1
        " 2>/dev/null || \
        konsole --title "摄像头摄像头" -e bash -c "
            echo '正在启动摄像头传感器模组...';
            cd '$CATKIN_WS_PATH';
            source devel/setup.sh;
            echo 'ROS环境已加载，启动摄像头传感器模组...';
            roslaunch xv_sdk xv_sdk.launch;
            echo '按任意键关闭此窗口...';
            read -n 1
        " 2>/dev/null || {
            print_warning "无法打开新终端窗口，将在当前终端中启动..."
            source devel/setup.sh
            roslaunch xv_sdk xv_sdk.launch
        }
    fi
    
    # 等待摄像头服务启动
    print_info "等待摄像头服务启动..."
    sleep 10
    
    # 检查摄像头话题是否可用
    local timeout=30
    local count=0
    
    while [ $count -lt $timeout ]; do
        if rostopic list 2>/dev/null | grep -q "/xv_sdk"; then
            print_success "摄像头摄像头服务已启动"
            return 0
        fi
        
        sleep 1
        count=$((count + 1))
        printf "\r等待摄像头服务... ($count/$timeout)"
    done
    
    echo ""
    print_error "摄像头服务启动超时"
    return 1
}

##############################################################################
# 动态检测摄像头设备序列号（支持多设备）
##############################################################################
detect_all_devices() {
    print_info "正在自动检测摄像头设备序列号..."
    
    # 获取设备序列号（过滤掉ROS参数服务话题）
    xvisio_SERIALS=($(rostopic list 2>/dev/null | grep -o '/xv_sdk/[^/]*' | cut -d'/' -f3 | grep -v -E '^(parameter_descriptions|parameter_updates|new_device)$' | sort -u))
    
    if [ ${#xvisio_SERIALS[@]} -eq 0 ]; then
        print_error "未能自动检测到摄像头设备序列号"
        print_error "请确保摄像头摄像头已正确启动并发布话题"
        return 1
    fi
    
    # 显示所有检测到的设备
    if [ ${#xvisio_SERIALS[@]} -eq 1 ]; then
        print_success "检测到 1 个设备: ${xvisio_SERIALS[0]}"
    else
        print_success "检测到 ${#xvisio_SERIALS[@]} 个设备:"
        for i in "${!xvisio_SERIALS[@]}"; do
            print_info "  设备 $((i+1)): ${xvisio_SERIALS[$i]}"
        done
    fi
    
    return 0
}

##############################################################################
# 启动夹具数据输出（支持多设备）
##############################################################################
start_clamp_service() {
    print_info "启动夹具数据输出..."
    
    # 等待一下确保摄像头服务完全就绪
    sleep 3
    
    # 动态检测所有设备序列号
    if ! detect_all_devices; then
        print_warning "无法检测到设备序列号，跳过夹具数据输出启动"
        return 0
    fi
    
    # 统计成功和失败的设备数量
    local success_count=0
    local skip_count=0
    local fail_count=0
    
    # 遍历所有检测到的设备
    for serial in "${xvisio_SERIALS[@]}"; do
        local clamp_service="/xv_sdk/${serial}/clamp/start"
        
        # 检查该设备的夹具服务是否可用
        if ! rosservice list 2>/dev/null | grep -q "$clamp_service"; then
            print_warning "设备 $serial: 夹具服务不可用，跳过"
            skip_count=$((skip_count + 1))
            continue
        fi
        
        # 调用夹具启动服务
        print_info "设备 $serial: 调用夹具服务 $clamp_service"
        if rosservice call "$clamp_service" 2>/dev/null; then
            print_success "设备 $serial: 夹具数据输出已启动"
            success_count=$((success_count + 1))
        else
            print_error "设备 $serial: 启动夹具数据输出失败"
            fail_count=$((fail_count + 1))
        fi
    done
    
    # 输出总结信息
    echo ""
    print_info "夹具服务启动总结:"
    print_info "  检测到设备: ${#xvisio_SERIALS[@]}"
    print_info "  启动成功: $success_count"
    if [ $skip_count -gt 0 ]; then
        print_info "  跳过: $skip_count (服务不可用)"
    fi
    if [ $fail_count -gt 0 ]; then
        print_info "  失败: $fail_count"
    fi
    
    # 只要有至少一个成功，就返回成功
    if [ $success_count -gt 0 ]; then
        return 0
    else
        return 1
    fi
}

##############################################################################
# 关闭夹具数据输出（支持多设备）
##############################################################################
stop_clamp_service() {
    print_info "关闭夹具数据输出..."
    

    # 动态检测所有设备序列号
    if ! detect_all_devices; then
        print_warning "无法检测到设备序列号，跳过夹具数据输出关闭"
        return 0
    fi
    
    # 统计成功和失败的设备数量
    local success_count=0
    local skip_count=0
    local fail_count=0
    
    # 遍历所有检测到的设备
    for serial in "${xvisio_SERIALS[@]}"; do
        local clamp_service="/xv_sdk/${serial}/clamp/stop"
        
        # 调用夹具关闭服务
        print_info "设备 $serial: 调用夹具服务 $clamp_service"
        if rosservice call "$clamp_service" 2>/dev/null; then
            print_success "设备 $serial: 夹具数据输出已关闭"
            success_count=$((success_count + 1))
        else
            print_error "设备 $serial: 关闭夹具数据输出失败"
            fail_count=$((fail_count + 1))
        fi
    done
    
    # 输出总结信息
    echo ""
    print_info "夹具服务关闭总结:"
    print_info "  检测到设备: ${#xvisio_SERIALS[@]}"
    print_info "  关闭成功: $success_count"
    if [ $skip_count -gt 0 ]; then
        print_info "  跳过: $skip_count (服务不可用)"
    fi
    if [ $fail_count -gt 0 ]; then
        print_info "  失败: $fail_count"
    fi
    
    # 只要有至少一个成功，就返回成功
    if [ $success_count -gt 0 ]; then
        return 0
    else
        return 1
    fi
}


##############################################################################
# 启动Vive发布器
##############################################################################
start_vive_publisher() {
    print_info "启动Vive发布器..."
    
    # 检查是否已有 vive_publisher 进程在运行
    if pgrep -f "vive_publisher.py" > /dev/null 2>&1; then
        print_warning "检测到已有 vive_publisher 进程在运行"
        local existing_count=$(pgrep -f "vive_publisher.py" | wc -l)
        print_warning "现有进程数: $existing_count"
        print_info "尝试清理现有进程..."
        pkill -f "vive_publisher.py" 2>/dev/null
        sleep 2
        pkill -9 -f "vive_publisher.py" 2>/dev/null
        sleep 1
    fi
    
    # 检查是否已有 vive_trackers_publisher 节点在 ROS 中运行
    if rosnode list 2>/dev/null | grep -q "vive_trackers_publisher"; then
        print_warning "检测到 ROS 中已有 vive_trackers_publisher 节点"
        print_info "清理旧节点..."
        rosnode kill /vive_trackers_publisher 2>/dev/null || true
        sleep 1
    fi
    
    # 检查Vive发布器文件是否存在
    if [ ! -f "$SCRIPT_DIR/vive_publisher.py" ]; then
        print_error "找不到vive_publisher.py文件"
        return 1
    fi
    
    # 切换到脚本目录并启动Vive发布器
    cd "$SCRIPT_DIR"
    
    if [ "$NO_GUI" = true ]; then
        # 后台模式
        python3 "$SCRIPT_DIR/vive_publisher.py" > "${SCRIPT_DIR}/vive.log" 2>&1 &
        local vive_pid=$!
        echo $vive_pid > "$VIVE_PID_FILE"
        print_info "Vive发布器已在后台启动，PID: $vive_pid"
    else
        # GUI模式：在新终端窗口启动
        gnome-terminal --title="Vive发布器" -- bash -c "
            cd '$SCRIPT_DIR';
            python3 '$SCRIPT_DIR/vive_publisher.py';
            echo '按任意键关闭此窗口...';
            read -n 1
        " 2>/dev/null || \
        xterm -title "Vive发布器" -e "
            cd '$SCRIPT_DIR';
            python3 '$SCRIPT_DIR/vive_publisher.py';
            echo '按任意键关闭此窗口...';
            read -n 1
        " 2>/dev/null || \
        konsole --title "Vive发布器" -e bash -c "
            cd '$SCRIPT_DIR';
            python3 '$SCRIPT_DIR/vive_publisher.py' ;
            echo '按任意键关闭此窗口...';
            read -n 1
        " 2>/dev/null || {
            print_warning "无法打开新终端窗口，将在当前终端中启动..."
            python3 "$SCRIPT_DIR/vive_publisher.py"
        }
    fi
    
    # 等待Vive发布器启动
    print_info "等待Vive发布器启动..."
    sleep 3
    
    # 检查Vive话题是否可用
    local timeout=30
    local count=0
    
    while [ $count -lt $timeout ]; do
        if rostopic list 2>/dev/null | grep -q "/vive/.*pose"; then
            print_success "Vive发布器已启动"
            return 0
        fi
        
        # 检查进程是否还在运行
        if [ -f "$VIVE_PID_FILE" ]; then
            local vive_pid=$(cat "$VIVE_PID_FILE")
            if ! ps -p $vive_pid > /dev/null 2>&1; then
                print_error "Vive发布器进程已退出 (PID: $vive_pid)"
                if [ -f "${SCRIPT_DIR}/vive.log" ]; then
                    print_info "查看vive.log获取错误信息："
                    tail -n 10 "${SCRIPT_DIR}/vive.log"
                fi
                return 1
            fi
        fi
        
        sleep 1
        count=$((count + 1))
        printf "\r等待Vive发布器... ($count/$timeout)"
    done
    
    echo ""
    print_warning "Vive发布器topic未检测到，可能仍在初始化中"
    return 0
}


##############################################################################
# 停止所有服务（增强版 - 完整清理）
##############################################################################
stop_all_services() {
    print_info "停止所有服务..."
    #0.关闭夹具服务
    stop_clamp_service
    
    # 1. 停止摄像头服务及其子进程
    if [ -f "$xvisio_PID_FILE" ]; then
        local xvisio_pid=$(cat "$xvisio_PID_FILE")
        if ps -p $xvisio_pid > /dev/null 2>&1; then
            print_info "停止摄像头服务 (PID: $xvisio_pid) 及其子进程..."
            # 先杀死整个进程树（所有子进程）
            pkill -P $xvisio_pid 2>/dev/null || true
            # 再杀死主进程
            kill $xvisio_pid 2>/dev/null || true
            sleep 2
            # 如果还在运行，强制终止
            if ps -p $xvisio_pid > /dev/null 2>&1; then
                print_warning "强制终止摄像头服务及其子进程..."
                pkill -9 -P $xvisio_pid 2>/dev/null || true
                kill -9 $xvisio_pid 2>/dev/null || true
            fi
        fi
        rm -f "$xvisio_PID_FILE"
    fi
    
    # 1.5 额外清理：杀死所有 xv_sdk 可执行文件进程（包括孤儿进程）
    print_info "清理所有 xv_sdk 可执行文件进程..."
    if pgrep -f "/xv_sdk/xv_sdk" > /dev/null 2>/dev/null; then
        pkill -f "/xv_sdk/xv_sdk" 2>/dev/null || true
        sleep 2
        # 强制杀死残留
        if pgrep -f "/xv_sdk/xv_sdk" > /dev/null 2>/dev/null; then
            print_warning "强制终止残留的 xv_sdk 进程..."
            pkill -9 -f "/xv_sdk/xv_sdk" 2>/dev/null || true
        fi
    fi
    
    # 2. 停止Vive服务及其子进程
    if [ -f "$VIVE_PID_FILE" ]; then
        local vive_pid=$(cat "$VIVE_PID_FILE")
        if ps -p $vive_pid > /dev/null 2>&1; then
            print_info "停止Vive发布器 (PID: $vive_pid) 及其子进程..."
            # 杀死子进程
            pkill -P $vive_pid 2>/dev/null || true
            # 杀死主进程
            kill $vive_pid 2>/dev/null || true
            sleep 2
            # 强制终止
            if ps -p $vive_pid > /dev/null 2>&1; then
                print_warning "强制终止Vive发布器及其子进程..."
                pkill -9 -P $vive_pid 2>/dev/null || true
                kill -9 $vive_pid 2>/dev/null || true
            fi
        fi
        rm -f "$VIVE_PID_FILE"
    fi
    
    # 3. 清理所有相关的 roslaunch 进程（包括孤儿进程）
    print_info "清理所有 roslaunch xv_sdk 进程..."
    pkill -f "roslaunch xv_sdk" 2>/dev/null || true
    sleep 1
    pkill -9 -f "roslaunch xv_sdk" 2>/dev/null || true
    
    # 4. 清理所有相关的 vive_publisher 进程
    print_info "清理所有 vive_publisher 进程..."
    pkill -f "vive_publisher.py" 2>/dev/null || true
    sleep 1
    pkill -9 -f "vive_publisher.py" 2>/dev/null || true
    
    # 5. 清理 xv_sdk 相关的 ROS 节点
    print_info "清理 xv_sdk 相关 ROS 节点..."
    for node in $(rosnode list 2>/dev/null | grep xv_sdk); do
        rosnode kill "$node" 2>/dev/null || true
    done
    
    # 6. 清理所有 vive 相关的 ROS 节点（包括匿名节点）
    print_info "清理 vive 相关 ROS 节点..."
    for node in $(rosnode list 2>/dev/null | grep -i vive); do
        rosnode kill "$node" 2>/dev/null || true
    done
    
    # 7. 额外清理：清理所有可能残留的匿名 vive_trackers_publisher 节点
    print_info "清理可能残留的匿名 vive 节点..."
    rosnode list 2>/dev/null | grep "vive_trackers_publisher" | while read -r node; do
        rosnode kill "$node" 2>/dev/null || true
    done
    sleep 1
    
    # 8. 使用 rosnode cleanup 清理所有无响应的死节点
    print_info "清理无响应的 ROS 死节点..."
    rosnode cleanup 2>/dev/null || true
    sleep 1
    
    # 9. 清理 PID 文件
    if [ -f "$PID_FILE" ]; then
        rm -f "$PID_FILE"
    fi
    
    # 10. 等待所有进程完全退出
    sleep 1
    
    # 11. 最后检查并显示残留进程/节点
    local remaining_xv=$(pgrep -f "xv_sdk" 2>/dev/null | wc -l)
    local remaining_vive=$(pgrep -f "vive_publisher" 2>/dev/null | wc -l)
    local remaining_vive_nodes=$(rosnode list 2>/dev/null | grep -i vive | wc -l)
    
    if [ $remaining_xv -gt 0 ]; then
        print_warning "检测到 $remaining_xv 个残留的 xv_sdk 进程"
        print_info "残留进程列表:"
        ps aux | grep xv_sdk | grep -v grep
    fi
    
    if [ $remaining_vive -gt 0 ]; then
        print_warning "检测到 $remaining_vive 个残留的 vive_publisher 进程"
        print_info "残留进程列表:"
        ps aux | grep vive_publisher | grep -v grep
    fi
    
    if [ $remaining_vive_nodes -gt 0 ]; then
        print_warning "检测到 $remaining_vive_nodes 个残留的 vive ROS 节点"
        print_info "残留节点列表:"
        rosnode list 2>/dev/null | grep -i vive
        print_info "提示: 这些可能是死节点，rosnode cleanup 已尝试清理"
    fi
    
    print_success "所有服务已停止"
}


##############################################################################
# 信号处理器
##############################################################################
signal_handler() {
    print_info "收到中断信号，正在退出..."
    EXIT_REQUESTED=true
    stop_all_services
    exit 0
}

# 注册信号处理器
trap signal_handler SIGINT SIGTERM

##############################################################################
# 主启动函数
##############################################################################
main_start() {
    print_info "开始启动服务..."
    
    # 检查环境
    if ! check_ros_environment; then
        exit 1
    fi
    
    if ! check_catkin_ws; then
        exit 1
    fi
    
    # 记录当前目录
    LOCAL_DIR=$(pwd)
    
    # 启动摄像头摄像头
    if ! start_xvisio_camera; then
        print_error "摄像头摄像头启动失败"
        exit 1
    fi
    
    # 等待ROS master和摄像头服务就绪
    if ! wait_for_rosmaster; then
        exit 1
    fi
    
    # 启动夹具数据输出
    start_clamp_service
    
    # 启动Vive发布器
    if ! start_vive_publisher; then
        print_error "Vive发布器启动失败"
        exit 1
    fi
    
    print_success "所有服务启动完成！"
    print_info "按Ctrl+C停止所有服务"
    
    # 保存当前进程PID
    echo $$ > "$PID_FILE"
    
    # 保持脚本运行并监控服务
    while ! $EXIT_REQUESTED; do
        sleep 5
        # 这里可以添加健康检查逻辑
    done
}

##############################################################################
# 主函数
##############################################################################
main() {
    # 默认参数
    NO_GUI=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            --no-gui)
                NO_GUI=true
                shift
                ;;
            *)
                print_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 启动所有服务
    main_start
}

# 运行主函数
main "$@"
