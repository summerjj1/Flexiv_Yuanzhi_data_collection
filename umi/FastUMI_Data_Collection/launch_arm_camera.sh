#!/bin/bash

##############################################################################
# 统一启动脚本 - 仅启动 摄像头 RGB 相机
# 功能：
# 1) 只启动 摄像头 的 RGB 图像发布，不启动 SLAM / ToF / 夹具 / Vive
# 2) 自动检测 ROS 环境与 catkin_ws
# 3) 支持 GUI/无界面两种模式
##############################################################################

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/launcher.log"
PID_FILE="$SCRIPT_DIR/launcher.pid"
xvisio_PID_FILE="$SCRIPT_DIR/xvisio.pid"

# 颜色定义
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

EXIT_REQUESTED=false

##############################################################################
# 帮助
##############################################################################
show_help() {
    echo "只启动 摄像头 RGB 相机"
    echo ""
    echo "用法: $0 [选项]"
    echo "  -h, --help     显示帮助"
    echo "  --no-gui       后台模式（不弹出新终端）"
    echo ""
    echo "示例:"
    echo "  $0              # GUI 模式启动 RGB"
    echo "  $0 --no-gui     # 后台模式启动 RGB"
}

##############################################################################
# 查找 catkin_ws
##############################################################################
find_catkin_ws() {
    if [ -d "catkin_ws" ]; then echo "catkin_ws"; return 0; fi
    if [ -d "../catkin_ws" ]; then echo "../catkin_ws"; return 0; fi
    if [ -d "$HOME/catkin_ws" ]; then echo "$HOME/catkin_ws"; return 0; fi
    local found_path
    found_path=$(find "$HOME" -maxdepth 3 -name "catkin_ws" -type d 2>/dev/null | head -1)
    if [ -n "$found_path" ]; then echo "$found_path"; return 0; fi
    return 1
}

##############################################################################
# 检查 ROS 环境
##############################################################################
check_ros_environment() {
    if ! command -v rospack >/dev/null 2>&1; then
        print_error "ROS 未安装或未在 PATH 中"
        return 1
    fi
    if [ -z "$ROS_DISTRO" ]; then
        print_warning "ROS_DISTRO 未设置，尝试 source /opt/ros/noetic/setup.bash ..."
        if [ -f "/opt/ros/noetic/setup.bash" ]; then
            # shellcheck source=/dev/null
            source /opt/ros/noetic/setup.bash
            print_info "已加载 ROS Noetic 环境"
        else
            print_error "找不到 /opt/ros/noetic/setup.bash"
            return 1
        fi
    fi
    return 0
}

##############################################################################
# 检查 catkin_ws
##############################################################################
check_catkin_ws() {
    print_info "查找 catkin_ws ..."
    CATKIN_WS_PATH=$(find_catkin_ws)
    if [ -z "$CATKIN_WS_PATH" ]; then
        print_error "找不到 catkin_ws，请确认以下位置之一存在："
        print_error "  $(pwd)/catkin_ws"
        print_error "  $(dirname "$(pwd)")/catkin_ws"
        print_error "  $HOME/catkin_ws"
        return 1
    fi
    print_success "找到 catkin_ws: $CATKIN_WS_PATH"
    if [ ! -f "$CATKIN_WS_PATH/devel/setup.sh" ]; then
        print_error "缺少 $CATKIN_WS_PATH/devel/setup.sh，请先编译工作空间"
        return 1
    fi
    return 0
}

##############################################################################
# 检查 ROS master
##############################################################################
check_rosmaster() {
    if pgrep -x "rosmaster" >/dev/null; then return 0; else return 1; fi
}

wait_for_rosmaster() {
    print_info "等待 ROS master ..."
    for i in $(seq 1 30); do
        if check_rosmaster; then
            print_success "ROS master 已运行，等待 2 秒以稳定..."
            sleep 2
            return 0
        fi
        sleep 1
        printf "\r等待中... (%d/30)" "$i"
    done
    echo ""
    print_error "等待 ROS master 超时"
    return 1
}

##############################################################################
# 启动 摄像头 仅 RGB
##############################################################################
start_xvisio_rgb_only() {
    print_info "启动 摄像头（仅 RGB）..."

    # 清理已存在的 xv_sdk 进程
    if pgrep -f "/xv_sdk/xv_sdk" >/dev/null 2>&1; then
        print_warning "检测到已有 xv_sdk 进程，进行清理..."
        pkill -f "/xv_sdk/xv_sdk" 2>/dev/null || true
        sleep 2
        pkill -9 -f "/xv_sdk/xv_sdk" 2>/dev/null || true
    fi

    cd "$CATKIN_WS_PATH" || exit 1
    # shellcheck source=/dev/null
    source devel/setup.sh

    # --- 方式 A：通过 roslaunch 参数禁用 SLAM/ToF，仅开 RGB ---
    #   xv_dev/slam/auto_start:=false  关闭 SLAM
    #   xv_dev/tof/auto_start:=false   关闭 ToF
    #   xv_dev/edge/auto_start:=false  关闭边缘模块(如无可忽略)
    LAUNCH_CMD_A=(roslaunch xv_sdk xv_sdk.launch
        xv_dev/slam/auto_start:=false
        xv_dev/tof/auto_start:=false
        xv_dev/edge/auto_start:=false
        #color/auto_start:=true
    )

    # --- 方式 B：在启动后通过 rosparam 显式关闭 ---
    POST_PARAMS_APPLY() {
        rosparam set /xv_sdk/xv_dev/slam/auto_start false 2>/dev/null || true
        rosparam set /xv_sdk/xv_dev/edge/auto_start false 2>/dev/null || true
        # 如果存在 ToF：
        rosparam set /xv_sdk/xv_dev/tof/auto_start false 2>/dev/null || true
    }

    if [ "$NO_GUI" = true ]; then
        nohup "${LAUNCH_CMD_A[@]}" > "$LOG_FILE" 2>&1 &
        local pid=$!
        echo $pid > "$xvisio_PID_FILE"
        print_info "xv_sdk 已后台启动 (PID: $pid)"
        # 稍等再推参数，避免节点还没起
        sleep 3
        POST_PARAMS_APPLY
    else
        # GUI 模式尽量开一个新终端
        if command -v gnome-terminal >/dev/null 2>&1; then
            gnome-terminal --title="摄像头 - RGB Only" -- bash -c "
                cd '$CATKIN_WS_PATH';
                source devel/setup.sh;
                echo '启动 xv_sdk（仅 RGB）...';
                ${LAUNCH_CMD_A[*]};
                read -n 1 -p '按任意键关闭窗口...'
            " 2>/dev/null || true
        elif command -v xterm >/dev/null 2>&1; then
            xterm -title "摄像头 - RGB Only" -e bash -c "
                cd '$CATKIN_WS_PATH';
                source devel/setup.sh;
                echo '启动 xv_sdk（仅 RGB）...';
                ${LAUNCH_CMD_A[*]};
                read -n 1 -p '按任意键关闭窗口...'
            " 2>/dev/null || true
        else
            print_warning "没有可用的终端程序，改为当前终端启动"
            "${LAUNCH_CMD_A[@]}"
        fi
    fi

    # 粗略健康检查：仅检查 RGB 话题是否出现
    print_info "等待 RGB 话题发布..."
    for i in $(seq 1 30); do
        if rostopic list 2>/dev/null | grep -q "/xv_sdk/.*/color_camera/image_color"; then
            print_success "检测到 RGB 话题 /xv_sdk/<serial>/color_camera/image_color"
            return 0
        fi
        sleep 1
        printf "\r等待中... (%d/30)" "$i"
    done
    echo ""
    print_warning "未检测到 RGB 话题（可能仍在初始化），请用 rostopic list 自查"
    return 0
}

##############################################################################
# 停止（仅清理 xv_sdk）
##############################################################################
stop_services() {
    print_info "停止 摄像头 RGB 服务..."
    if [ -f "$xvisio_PID_FILE" ]; then
        local pid
        pid=$(cat "$xvisio_PID_FILE")
        if ps -p "$pid" >/dev/null 2>&1; then
            print_info "终止进程 (PID: $pid)"
            pkill -P "$pid" 2>/dev/null || true
            kill "$pid" 2>/dev/null || true
            sleep 2
            if ps -p "$pid" >/dev/null 2>&1; then
                print_warning "强制终止 xv_sdk ..."
                pkill -9 -P "$pid" 2>/dev/null || true
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
        rm -f "$xvisio_PID_FILE"
    fi

    # 兜底：杀掉残留 roslaunch xv_sdk / xv_sdk 可执行
    pkill -f "roslaunch xv_sdk" 2>/dev/null || true
    sleep 1
    pkill -9 -f "roslaunch xv_sdk" 2>/dev/null || true
    pkill -f "/xv_sdk/xv_sdk" 2>/dev/null || true
    sleep 1
    pkill -9 -f "/xv_sdk/xv_sdk" 2>/dev/null || true

    # 清理相关节点
    rosnode list 2>/dev/null | grep -E "xv_sdk" | while read -r node; do
        rosnode kill "$node" 2>/dev/null || true
    done
    rosnode cleanup 2>/dev/null || true

    rm -f "$PID_FILE"
    print_success "已停止"
}

##############################################################################
# 信号处理
##############################################################################
signal_handler() {
    print_info "收到中断信号，准备退出..."
    EXIT_REQUESTED=true
    stop_services
    exit 0
}
trap signal_handler SIGINT SIGTERM

##############################################################################
# 主启动
##############################################################################
main_start() {
    print_info "开始启动（仅 RGB）..."
    check_ros_environment || exit 1
    check_catkin_ws || exit 1

    # 记录 PID
    echo $$ > "$PID_FILE"

    # 确保 roscore 在
    if ! check_rosmaster; then
        print_warning "ROS master 未运行，将由 roslaunch 自动拉起（或你可先开 roscore）"
    fi

    start_xvisio_rgb_only || {
        print_error "启动失败"
        exit 1
    }

    wait_for_rosmaster || exit 1

    print_success "摄像头 RGB 已启动。按 Ctrl+C 停止。"
    while ! $EXIT_REQUESTED; do sleep 5; done
}

##############################################################################
# 入口
##############################################################################
main() {
    NO_GUI=false
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help) show_help; exit 0 ;;
            --no-gui)  NO_GUI=true; shift ;;
            *) print_error "未知参数: $1"; show_help; exit 1 ;;
        esac
    done
    main_start
}

main "$@"
