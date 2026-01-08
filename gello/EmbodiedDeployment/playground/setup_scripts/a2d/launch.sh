# 1. source机器人环境
conda activate lerobot
cd /root/workspace/A2D/a2d_sdk
source env.sh
export PYTHONPATH=/root/workspace/A2D:$PYTHONPATH

# 2. 调整机器人至控制模式，输入指令以后会听到机器人关节响声，同时控制台输出如图1 robot ready/camera OK字样
robot-service -s -c ./conf/copilot.pbtxt

# (Optional)2. 如果启动机器人控制失败，先切换至休闲模式再切换回来
robot-service -s -c ./conf/idle.pbtxt

# 3. 测试机器人是否可控，以头部为例，初始是0，10；输入10，10，机器人头部会有明显转动
robot-controller
head 10,10

# 4. 启动A2D示例程序
python playground/my_robot/a2d_dual_arm.py