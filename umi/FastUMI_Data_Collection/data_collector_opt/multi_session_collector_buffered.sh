#!/bin/bash
# -*- coding: utf-8 -*-
#
# 多会话摄像头数据采集脚本（Buffered版本）
# 支持连续多次数据采集，支持单设备/双设备模式
# 自动从config.json读取设备配置
#

# 默认配置参数
DEFAULT_TOTAL_SESSIONS=5
DEFAULT_INTERVAL_SECONDS=3
DEFAULT_PYTHON_SCRIPT="./single_session_data_collector_buffered.py"
DEFAULT_TOF_MODE="off"
DEFAULT_CONFIG_PATH="../start_process/config.json"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 全局变量
TOTAL_SESSIONS=0
INTERVAL_SECONDS=0
PYTHON_SCRIPT=""
MAIN_OUTPUT_DIR=""
SUMMARY_FILE=""
CURRENT_SESSION=0
SHOULD_STOP=false
TOF_MODE=""
CONFIG_PATH=""
IS_SINGLE_DEVICE=true
DEVICE_COUNT=1
SESSION_OUTPUT_DIRS=()  # 记录每次session生成的所有目录

# 显示帮助信息
show_help() {
    echo "多会话摄像头数据采集脚本（Buffered版本）"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -s, --sessions NUM      采集会话数量 (默认: $DEFAULT_TOTAL_SESSIONS)"
    echo "  -i, --interval SECONDS  会话间隔时间(秒) (默认: $DEFAULT_INTERVAL_SECONDS)"
    echo "  -p, --python-script PATH Python脚本路径 (默认: $DEFAULT_PYTHON_SCRIPT)"
    echo "  -c, --config PATH       配置文件路径 (默认: $DEFAULT_CONFIG_PATH)"
    echo "  -o, --output DIR        摘要文件输出目录 (默认: ./multi_sessions_TIMESTAMP)"
    echo "  --tof [on|off]          是否采集ToF点云 (默认: $DEFAULT_TOF_MODE)"
    echo "  -h, --help              显示此帮助信息"
    echo ""
    echo "特性:"
    echo "  • 自动从config.json读取设备配置"
    echo "  • 支持单设备/双设备模式自动识别"
    echo "  • 每次采集使用独立Python进程，避免内存泄漏"
    echo "  • Python脚本自动生成带时间戳的目录"
    echo ""
    echo "控制说明:"
    echo "  • 在采集过程中按 Ctrl+C 可以停止当前采集"
    echo "  • 每次采集完成后会询问是否继续下一次采集"
    echo "  • 在任何时候按 Ctrl+C 都可以优雅地停止程序"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认参数"
    echo "  $0 -s 10 -i 5                         # 采集10次，间隔5秒"
    echo "  $0 --sessions 3 --tof on              # 采集3次，启用ToF"
}

# 打印带颜色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_success() {
    print_message $GREEN "$1"
}

print_error() {
    print_message $RED "$1"
}

print_warning() {
    print_message $YELLOW "$1"
}

print_info() {
    print_message $BLUE "$1"
}

print_highlight() {
    print_message $CYAN "$1"
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--sessions)
                TOTAL_SESSIONS="$2"
                shift 2
                ;;
            -i|--interval)
                INTERVAL_SECONDS="$2"
                shift 2
                ;;
            -p|--python-script)
                PYTHON_SCRIPT="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_PATH="$2"
                shift 2
                ;;
            -o|--output)
                MAIN_OUTPUT_DIR="$2"
                shift 2
                ;;
            --tof)
                TOF_MODE="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 设置默认值
    if [ "$TOTAL_SESSIONS" -eq 0 ]; then
        TOTAL_SESSIONS="$DEFAULT_TOTAL_SESSIONS"
    fi
    
    if [ "$INTERVAL_SECONDS" -eq 0 ]; then
        INTERVAL_SECONDS="$DEFAULT_INTERVAL_SECONDS"
    fi
    
    if [ -z "$PYTHON_SCRIPT" ]; then
        PYTHON_SCRIPT="$DEFAULT_PYTHON_SCRIPT"
    fi
    
    if [ -z "$CONFIG_PATH" ]; then
        CONFIG_PATH="$DEFAULT_CONFIG_PATH"
    fi
    
    if [ -z "$TOF_MODE" ]; then
        TOF_MODE="$DEFAULT_TOF_MODE"
    fi
    
    if [ "$TOF_MODE" != "on" ] && [ "$TOF_MODE" != "off" ]; then
        print_warning "无效的 ToF 模式: $TOF_MODE，回退为默认值: $DEFAULT_TOF_MODE"
        TOF_MODE="$DEFAULT_TOF_MODE"
    fi
    
    # 默认输出目录（带时间戳）
    if [ -z "$MAIN_OUTPUT_DIR" ]; then
        local timestamp=$(date +"%Y%m%d_%H%M%S")
        MAIN_OUTPUT_DIR="./multi_sessions_${timestamp}"
    fi
}

# 使用Python读取JSON配置的辅助函数
json_get() {
    local json_file=$1
    local json_path=$2
    
    python3 -c "
import json
import sys

try:
    with open('$json_file', 'r') as f:
        data = json.load(f)
    
    # 解析路径（如 'single_device' 或 'devices.device_0.xv_serial'）
    keys = '$json_path'.split('.')
    value = data
    for key in keys:
        value = value[key]
    
    print(value)
except Exception as e:
    sys.stderr.write(f'Error: {e}\n')
    sys.exit(1)
"
}

# 读取并解析config.json
load_config() {
    print_info "加载配置文件: $CONFIG_PATH"
    
    # 检查配置文件是否存在
    if [ ! -f "$CONFIG_PATH" ]; then
        print_error "错误: 找不到配置文件 $CONFIG_PATH"
        print_info ""
        print_info "请先运行以下命令之一："
        print_info "  1. 启动完整系统（推荐）:"
        print_info "     cd ../start_process && ./unified_launcher.sh"
        print_info "  2. 仅运行设备配对:"
        print_info "     cd ../start_process && python3 device_pairing.py"
        exit 1
    fi
    
    # 使用Python读取single_device字段
    IS_SINGLE_DEVICE=$(json_get "$CONFIG_PATH" "single_device")
    
    if [ "$IS_SINGLE_DEVICE" = "True" ]; then
        DEVICE_COUNT=1
        print_success "✓ 配置模式: 单设备"
        
        # 读取设备信息
        local xv_serial=$(json_get "$CONFIG_PATH" "devices.device_0.xv_serial")
        local vive_serial=$(json_get "$CONFIG_PATH" "devices.device_0.vive_serial")
        local label=$(json_get "$CONFIG_PATH" "devices.device_0.label")
        
        print_info "  设备标签: $label"
        print_info "  XV序列号: $xv_serial"
        print_info "  Vive序列号: $vive_serial"
        
    elif [ "$IS_SINGLE_DEVICE" = "False" ]; then
        DEVICE_COUNT=2
        print_success "✓ 配置模式: 双设备"
        
        # 读取两个设备的信息
        for device_idx in 0 1; do
            local xv_serial=$(json_get "$CONFIG_PATH" "devices.device_${device_idx}.xv_serial")
            local vive_serial=$(json_get "$CONFIG_PATH" "devices.device_${device_idx}.vive_serial")
            local label=$(json_get "$CONFIG_PATH" "devices.device_${device_idx}.label")
            
            print_info "  设备 $((device_idx+1)) [$label]:"
            print_info "    XV序列号: $xv_serial"
            print_info "    Vive序列号: $vive_serial"
        done
    else
        print_error "错误: 配置文件格式无效，single_device字段必须为true或false"
        exit 1
    fi
}

# 检查环境
check_environment() {
    print_info "检查运行环境..."
    
    # 检查Python脚本是否存在
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_error "错误: 找不到Python脚本 $PYTHON_SCRIPT"
        print_info "请确保脚本路径正确，或使用 -p 参数指定正确的路径"
        exit 1
    fi
    
    # 检查ROS是否运行
    if ! rostopic list > /dev/null 2>&1; then
        print_error "错误: ROS未运行，请先启动ROS"
        print_info "可以运行: roscore"
        exit 1
    fi
    
    # 检查必要的ROS话题是否存在（根据设备数量）
    print_info "检查ROS话题..."
    
    if [ "$IS_SINGLE_DEVICE" = "True" ]; then
        check_device_topics 0
    else
        check_device_topics 0
        check_device_topics 1
    fi
    
    print_success "环境检查完成"
}

# 检查单个设备的话题
check_device_topics() {
    local device_idx=$1
    local xv_serial=$(json_get "$CONFIG_PATH" "devices.device_${device_idx}.xv_serial")
    local vive_serial=$(json_get "$CONFIG_PATH" "devices.device_${device_idx}.vive_serial")
    
    # 格式化vive_serial（LHR-XXX -> LHR_XXX）
    local vive_serial_safe="${vive_serial//-/_}"
    
    local topics=(
        "/xv_sdk/${xv_serial}/color_camera/image_color"
        "/xv_sdk/${xv_serial}/slam/pose"
        "/vive/${vive_serial_safe}/pose"
        "/xv_sdk/${xv_serial}/clamp/Data"
    )
    
    if [ "$TOF_MODE" = "on" ]; then
        topics+=("/xv_sdk/${xv_serial}/tof_camera/point_cloud")
    fi
    
    local missing_topics=()
    for topic in "${topics[@]}"; do
        if ! rostopic list | grep -q "^${topic}$"; then
            missing_topics+=("$topic")
        fi
    done
    
    if [ ${#missing_topics[@]} -gt 0 ]; then
        print_warning "设备 $((device_idx+1)) 以下话题不存在:"
        for topic in "${missing_topics[@]}"; do
            echo "  - $topic"
        done
        print_warning "请确保所有传感器都已正确连接和启动"
        read -p "是否继续？(y/n): " continue_choice
        if [[ $continue_choice != "y" && $continue_choice != "Y" ]]; then
            exit 1
        fi
    fi
}

# 初始化输出目录
init_output_directory() {
    print_info "初始化输出目录..."
    
    # 创建主输出目录
    mkdir -p "$MAIN_OUTPUT_DIR"
    
    # 创建采集摘要文件
    SUMMARY_FILE="$MAIN_OUTPUT_DIR/collection_summary.txt"
    
    # 写入摘要文件头部
    cat > "$SUMMARY_FILE" << EOF
========================================
多会话摄像头数据采集摘要（Buffered版本）
========================================
开始时间: $(date)
设备模式: $([ "$IS_SINGLE_DEVICE" = "True" ] && echo "单设备" || echo "双设备")
设备数量: $DEVICE_COUNT
计划会话数: $TOTAL_SESSIONS
会话间隔: ${INTERVAL_SECONDS}秒
摘要目录: $MAIN_OUTPUT_DIR
ToF模式: $TOF_MODE
Python脚本: $PYTHON_SCRIPT
配置文件: $CONFIG_PATH
========================================

EOF
    
    # 添加设备详细信息
    if [ "$IS_SINGLE_DEVICE" = "True" ]; then
        {
            echo "设备信息:"
            echo "  XV序列号: $(json_get "$CONFIG_PATH" "devices.device_0.xv_serial")"
            echo "  Vive序列号: $(json_get "$CONFIG_PATH" "devices.device_0.vive_serial")"
            echo "  设备标签: $(json_get "$CONFIG_PATH" "devices.device_0.label")"
            echo ""
        } >> "$SUMMARY_FILE"
    else
        {
            echo "设备信息:"
            for device_idx in 0 1; do
                echo "  设备 $((device_idx+1)):"
                echo "    XV序列号: $(json_get "$CONFIG_PATH" "devices.device_${device_idx}.xv_serial")"
                echo "    Vive序列号: $(json_get "$CONFIG_PATH" "devices.device_${device_idx}.vive_serial")"
                echo "    设备标签: $(json_get "$CONFIG_PATH" "devices.device_${device_idx}.label")"
            done
            echo ""
        } >> "$SUMMARY_FILE"
    fi
    
    print_success "输出目录初始化完成: $MAIN_OUTPUT_DIR"
}

# 获取会话的输出目录列表（根据设备模式）
get_session_output_dirs() {
    local session_base_dir=$1
    local dirs=()
    
    if [ "$IS_SINGLE_DEVICE" = "True" ]; then
        # 单设备模式：直接返回 session_XXX 目录
        dirs+=("$session_base_dir")
    else
        # 双设备模式：返回 session_XXX 下的两个设备子目录
        # 需要从配置中读取设备标签和序列号
        for device_idx in 0 1; do
            local label=$(json_get "$CONFIG_PATH" "devices.device_${device_idx}.label")
            local xv_serial=$(json_get "$CONFIG_PATH" "devices.device_${device_idx}.xv_serial")
            local device_subdir="${label}_${xv_serial}"
            dirs+=("${session_base_dir}/${device_subdir}")
        done
    fi
    
    # 返回结果
    printf '%s\n' "${dirs[@]}"
}

# 验证单个目录的数据
validate_single_directory() {
    local session_dir=$1
    local device_label=$2
    
    if [ ! -d "$session_dir" ]; then
        print_error "  [$device_label] 目录不存在: $session_dir"
        return 1
    fi
    
    local validation_passed=true
    local missing_files=()
    
    # 检查关键文件是否存在
    local required_files=(
        "RGB_Images/video.mp4"
        "RGB_Images/timestamps.csv"
        "SLAM_Poses/slam_raw.txt"
        "Vive_Poses/vive_data_tum.txt"
        "Vive_Poses/offset_info.txt"
        "Clamp_Data/clamp_data_tum.txt"
        "Merged_Trajectory/merged_trajectory.txt"
        "Merged_Trajectory/merge_stats.txt"
    )
    
    if [ "$TOF_MODE" = "on" ]; then
        required_files+=("ToF_PointClouds/timestamps.csv")
    fi
    
    for file in "${required_files[@]}"; do
        local full_path="$session_dir/$file"
        if [ ! -e "$full_path" ]; then
            missing_files+=("$file")
            validation_passed=false
        fi
    done
    
    # 检查ToF点云文件数量（仅当开启ToF时）
    if [ "$TOF_MODE" = "on" ]; then
        local tof_dir="$session_dir/ToF_PointClouds/PointClouds"
        if [ -d "$tof_dir" ]; then
            local pcd_count=$(find "$tof_dir" -name "*.pcd" 2>/dev/null | wc -l)
            if [ "$pcd_count" -eq 0 ]; then
                missing_files+=("ToF_PointClouds/PointClouds/*.pcd")
                validation_passed=false
            fi
        else
            missing_files+=("ToF_PointClouds/PointClouds/")
            validation_passed=false
        fi
    fi
    
    if [ "$validation_passed" = true ]; then
        return 0
    else
        print_error "  [$device_label] 数据验证失败，缺少以下文件:"
        for file in "${missing_files[@]}"; do
            echo "    - $file"
        done
        return 1
    fi
}

# 验证会话数据（支持单/双设备）
validate_session_data() {
    local session_num=$1
    shift
    local output_dirs=("$@")
    
    print_info "验证会话 $session_num 数据完整性..."
    
    if [ ${#output_dirs[@]} -eq 0 ]; then
        print_error "  错误: 未找到输出目录"
        return 1
    fi
    
    local all_valid=true
    
    if [ "$IS_SINGLE_DEVICE" = "True" ]; then
        # 单设备模式：应该只有1个目录
        if [ ${#output_dirs[@]} -ne 1 ]; then
            print_warning "  警告: 单设备模式但检测到 ${#output_dirs[@]} 个目录"
        fi
        
        if ! validate_single_directory "${output_dirs[0]}" "单设备"; then
            all_valid=false
        fi
    else
        # 双设备模式：应该有2个目录
        if [ ${#output_dirs[@]} -ne 2 ]; then
            print_warning "  警告: 双设备模式但检测到 ${#output_dirs[@]} 个目录"
        fi
        
        for idx in "${!output_dirs[@]}"; do
            local device_label="设备$((idx+1))"
            if ! validate_single_directory "${output_dirs[$idx]}" "$device_label"; then
                all_valid=false
            fi
        done
    fi
    
    if [ "$all_valid" = true ]; then
        print_success "会话 $session_num 数据验证通过"
        return 0
    else
        return 1
    fi
}

# 显示进度
show_progress() {
    local current=$1
    local total=$2
    local percentage=$((current * 100 / total))
    local bar_length=50
    local filled_length=$((current * bar_length / total))
    
    printf "\r进度: ["
    for ((i=0; i<filled_length; i++)); do printf "█"; done
    for ((i=filled_length; i<bar_length; i++)); do printf "░"; done
    printf "] %d/%d (%d%%) " $current $total $percentage
}

# 执行单次采集
execute_single_session() {
    local session_num=$1
    
    print_info ""
    print_highlight "========================================"
    print_highlight "开始第 $session_num/$TOTAL_SESSIONS 次采集"
    print_highlight "========================================"
    
    # 构造会话输出目录（统一使用 session_XXX 格式）
    local session_base_dir="$MAIN_OUTPUT_DIR/session_$(printf "%03d" $session_num)"
    
    print_info "会话输出目录: $session_base_dir"
    
    # 记录开始时间
    local start_time=$(date)
    local start_timestamp=$(date +%s)
    
    # 创建临时日志文件
    local temp_log=$(mktemp)
    
    # 更新摘要文件
    {
        echo ""
        echo "=========================================="
        echo "会话 $session_num 信息:"
        echo "=========================================="
        echo "  开始时间: $start_time"
        echo "  输出目录: $session_base_dir"
    } >> "$SUMMARY_FILE"
    
    # 执行Python脚本（通过 --output 参数指定输出目录）
    print_info "启动数据采集进程..."
    print_info "命令: python3 -u $PYTHON_SCRIPT --output \"$session_base_dir\" --tof $TOF_MODE"
    
    # 使用 -u 参数强制 Python 无缓冲输出，确保实时显示进度
    if python3 -u "$PYTHON_SCRIPT" --output "$session_base_dir" --tof "$TOF_MODE" 2>&1 | tee "$temp_log"; then
        local end_time=$(date)
        local end_timestamp=$(date +%s)
        local duration=$((end_timestamp - start_timestamp))
        
        print_success "第 $session_num 次采集成功完成 (耗时: ${duration}秒)"
        
        # 获取输出目录列表（根据设备模式）
        mapfile -t output_dirs < <(get_session_output_dirs "$session_base_dir")
        
        # 记录所有输出目录
        print_success "数据目录结构:"
        if [ "$IS_SINGLE_DEVICE" = "True" ]; then
            echo "  $session_base_dir"
        else
            echo "  $session_base_dir/"
            for dir in "${output_dirs[@]}"; do
                echo "    ├─ $(basename "$dir")"
            done
        fi
        
        # 保存到全局列表
        for dir in "${output_dirs[@]}"; do
            SESSION_OUTPUT_DIRS+=("$dir")
        done
        
        # 验证数据
        if validate_session_data "$session_num" "${output_dirs[@]}"; then
            # 更新摘要文件 - 成功
            {
                echo "  结束时间: $end_time"
                echo "  持续时间: ${duration}秒"
                echo "  状态: 成功"
                echo "  数据验证: 通过"
                if [ "$IS_SINGLE_DEVICE" = "True" ]; then
                    echo "  数据目录: $session_base_dir"
                else
                    echo "  数据目录: $session_base_dir/"
                    for dir in "${output_dirs[@]}"; do
                        echo "    ├─ $(basename "$dir")"
                    done
                fi
            } >> "$SUMMARY_FILE"
            
            rm -f "$temp_log"
            return 0
        else
            # 更新摘要文件 - 验证失败
            {
                echo "  结束时间: $end_time"
                echo "  持续时间: ${duration}秒"
                echo "  状态: 验证失败"
                if [ "$IS_SINGLE_DEVICE" = "True" ]; then
                    echo "  数据目录: $session_base_dir"
                else
                    echo "  数据目录: $session_base_dir/"
                    for dir in "${output_dirs[@]}"; do
                        echo "    ├─ $(basename "$dir")"
                    done
                fi
            } >> "$SUMMARY_FILE"
            
            rm -f "$temp_log"
            return 1
        fi
    else
        local end_time=$(date)
        print_error "第 $session_num 次采集失败"
        
        # 更新摘要文件 - 失败
        {
            echo "  结束时间: $end_time"
            echo "  状态: 失败"
        } >> "$SUMMARY_FILE"
        
        rm -f "$temp_log"
        return 1
    fi
}

# 显示会话间隔倒计时
show_countdown() {
    local seconds=$1
    for ((i=seconds; i>0; i--)); do
        if [ "$SHOULD_STOP" = true ]; then
            return 1
        fi
        printf "\r下次采集倒计时: %d秒 (按Ctrl+C中断) " $i
        sleep 1
    done
    printf "\r                                                  \r"
}

# 主采集循环
run_collection_loop() {
    print_info ""
    print_highlight "========================================"
    print_highlight "开始多会话数据采集"
    print_highlight "========================================"
    print_info "设备模式: $([ "$IS_SINGLE_DEVICE" = "True" ] && echo "单设备" || echo "双设备 (${DEVICE_COUNT}个设备)")"
    print_info "计划采集会话数: $TOTAL_SESSIONS"
    print_info "会话间隔: ${INTERVAL_SECONDS}秒"
    print_info "摘要目录: $MAIN_OUTPUT_DIR"
    print_info "ToF模式: $TOF_MODE"
    print_highlight "========================================"
    print_info ""
    print_info "控制提示:"
    print_info "  • 在采集过程中按 Ctrl+C 停止当前采集"
    print_info "  • 每次采集完成后会询问是否继续"
    print_info "  • 程序会优雅地处理中断信号"
    print_highlight "========================================"
    
    local successful_sessions=0
    local failed_sessions=0
    
    for ((session=1; session<=TOTAL_SESSIONS; session++)); do
        # 检查是否应该停止
        if [ "$SHOULD_STOP" = true ]; then
            print_warning "用户请求停止，退出采集循环"
            break
        fi
        
        CURRENT_SESSION=$session
        
        # 显示进度
        show_progress $((session-1)) $TOTAL_SESSIONS
        echo ""
        
        # 执行采集
        if execute_single_session $session; then
            ((successful_sessions++))
        else
            ((failed_sessions++))
            print_warning "会话 $session 采集失败"
            
            # 询问是否继续
            if [ $session -lt $TOTAL_SESSIONS ]; then
                read -p "是否继续下一组采集？(y/n/s=跳过): " continue_choice
                case $continue_choice in
                    "s"|"S"|"skip"|"跳过")
                        print_info "跳过当前会话，继续下一组"
                        ;;
                    "n"|"N"|"no"|"否")
                        print_info "用户选择停止采集"
                        break
                        ;;
                    *)
                        print_info "继续下一组采集"
                        ;;
                esac
            fi
        fi
        
        # 会话间隔
        if [ $session -lt $TOTAL_SESSIONS ]; then
            print_info ""
            print_success "第 $session 次采集完成。"
            print_info "选择操作："
            print_info "  [c] 继续下一次采集 (默认)"
            print_info "  [s] 跳过剩余采集"
            print_info "  [q] 退出程序"
            print_info ""
            
            # 等待用户输入
            read -p "请选择 (c/s/q): " user_choice
            
            case $user_choice in
                "s"|"S"|"skip"|"跳过")
                    print_warning "用户选择跳过剩余采集"
                    break
                    ;;
                "q"|"Q"|"quit"|"exit"|"退出")
                    print_warning "用户选择退出程序"
                    SHOULD_STOP=true
                    break
                    ;;
                "c"|"C"|"continue"|"继续"|"")
                    print_info "继续下一次采集..."
                    print_info "等待 ${INTERVAL_SECONDS} 秒后开始..."
                    if ! show_countdown $INTERVAL_SECONDS; then
                        print_warning "用户请求停止采集"
                        break
                    fi
                    ;;
                *)
                    print_warning "无效选择，默认继续"
                    print_info "等待 ${INTERVAL_SECONDS} 秒后开始..."
                    if ! show_countdown $INTERVAL_SECONDS; then
                        print_warning "用户请求停止采集"
                        break
                    fi
                    ;;
            esac
        fi
    done
    
    # 显示最终进度
    show_progress $TOTAL_SESSIONS $TOTAL_SESSIONS
    echo ""
    
    # 完成总结
    print_info ""
    print_highlight "========================================"
    print_highlight "所有采集任务完成"
    print_highlight "========================================"
    print_info "成功会话数: $successful_sessions"
    print_info "失败会话数: $failed_sessions"
    print_info "总计划会话数: $TOTAL_SESSIONS"
    print_info "实际完成会话数: $((successful_sessions + failed_sessions))"
    
    if [ $((successful_sessions + failed_sessions)) -gt 0 ]; then
        local success_rate=$((successful_sessions * 100 / (successful_sessions + failed_sessions)))
        print_info "成功率: ${success_rate}%"
    fi
    
    # 更新摘要文件
    {
        echo ""
        echo "=========================================="
        echo "采集完成总结:"
        echo "=========================================="
        echo "完成时间: $(date)"
        echo "成功会话数: $successful_sessions"
        echo "失败会话数: $failed_sessions"
        echo "总计划会话数: $TOTAL_SESSIONS"
        echo "实际完成会话数: $((successful_sessions + failed_sessions))"
        if [ $((successful_sessions + failed_sessions)) -gt 0 ]; then
            echo "成功率: $((successful_sessions * 100 / (successful_sessions + failed_sessions)))%"
        fi
        echo ""
        echo "数据目录结构:"
        echo "$MAIN_OUTPUT_DIR/"
        # 列出所有 session_XXX 目录
        for ((i=1; i<=successful_sessions+failed_sessions; i++)); do
            session_dir="session_$(printf "%03d" $i)"
            if [ -d "$MAIN_OUTPUT_DIR/$session_dir" ]; then
                if [ "$IS_SINGLE_DEVICE" = "True" ]; then
                    echo "├─ $session_dir/"
                else
                    echo "├─ $session_dir/"
                    # 列出设备子目录
                    for subdir in "$MAIN_OUTPUT_DIR/$session_dir"/*; do
                        if [ -d "$subdir" ]; then
                            echo "│  ├─ $(basename "$subdir")/"
                        fi
                    done
                fi
            fi
        done
        echo "└─ collection_summary.txt"
    } >> "$SUMMARY_FILE"
    
    print_success "采集摘要已保存到: $SUMMARY_FILE"
    print_success "所有会话数据已保存在: $MAIN_OUTPUT_DIR"
}

# 信号处理
cleanup_on_exit() {
    print_warning ""
    print_warning "收到中断信号，正在清理..."
    SHOULD_STOP=true
    
    if [ $CURRENT_SESSION -gt 0 ]; then
        print_info "当前正在进行第 $CURRENT_SESSION 次采集"
        print_info "正在等待当前采集完成..."
        
        # 更新摘要文件
        {
            echo ""
            echo "=========================================="
            echo "采集被用户中断"
            echo "=========================================="
            echo "中断时间: $(date)"
            echo "中断时会话: $CURRENT_SESSION/$TOTAL_SESSIONS"
        } >> "$SUMMARY_FILE"
    fi
    
    print_info "清理完成"
    exit 1
}

# 主函数
main() {
    # 设置信号处理
    trap cleanup_on_exit SIGINT SIGTERM
    
    # 解析命令行参数
    parse_arguments "$@"
    
    # 加载配置
    load_config
    
    # 检查环境
    check_environment
    
    # 初始化输出目录
    init_output_directory
    
    # 运行采集循环
    run_collection_loop
    
    print_success ""
    print_success "多会话数据采集完成！"
    print_info "摘要文件: $SUMMARY_FILE"
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

