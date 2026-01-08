"""Robot使用示例。

展示如何使用Robot类进行基本的机器人操作。
"""

import time

from unit_test.mock.mock_robot import MockRobot, SingleArmMockRobot


def example_basic_usage():
    """基础使用示例：创建、初始化、获取观察、执行动作。"""
    print("=" * 50)
    print("示例1: 基础使用")
    print("=" * 50)

    # 创建单臂机器人
    robot = SingleArmMockRobot()

    # 使用上下文管理器自动初始化和清理
    with robot:
        # 获取观察数据
        observation = robot.get()
        print(f"观察数据键: {list(observation.keys())}")
        print(f"控制器数据: {list(observation['controllers'].keys())}")
        print(f"传感器数据: {list(observation['sensors'].keys())}")

        # 执行动作
        move_data = {
            "left_arm": {
                "action_type": "joint",
                "action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            }
        }
        robot.move(move_data)
        print("已执行移动命令")

        # 再次获取观察，查看状态变化
        observation_after = robot.get()
        print(f"执行后观察数据已更新")

        # 重置机器人
        robot.reset()
        print("机器人已重置")


def example_multi_arm_robot():
    """多臂机器人使用示例。"""
    print("\n" + "=" * 50)
    print("示例2: 多臂机器人")
    print("=" * 50)

    # 创建双臂机器人，每个手臂7个关节，2个摄像头
    robot = MockRobot(num_arms=2, num_joints_per_arm=7, num_cameras=2)

    # 手动初始化和清理
    robot.set_up()
    try:
        # 获取观察数据
        observation = robot.get()
        print(f"机器人名称: {robot.name}")
        print(f"控制器: {list(observation['controllers'].keys())}")
        print(f"传感器: {list(observation['sensors'].keys())}")

        # 同时控制两个手臂
        move_data = {
            "arm_0": {
                "action_type": "joint",
                "action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            },
            "arm_1": {
                "action_type": "joint",
                "action": [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7],
            },
        }
        robot.move(move_data)
        print("已同时控制两个手臂")

        # 获取摄像头数据
        for camera_name, camera_data in observation["sensors"].items():
            if camera_data is not None:
                print(
                    f"摄像头 {camera_name} 数据形状: {camera_data.get('rgb', {}).shape if isinstance(camera_data.get('rgb'), type(camera_data)) else 'N/A'}"
                )

    finally:
        robot.close()
        print("机器人已关闭")


def example_control_loop():
    """控制循环示例：模拟感知-决策-执行循环。"""
    print("\n" + "=" * 50)
    print("示例3: 控制循环")
    print("=" * 50)

    robot = SingleArmMockRobot()

    with robot:
        # 模拟控制循环
        num_steps = 5
        for step in range(num_steps):
            # 1. 获取观察
            observation = robot.get()
            print(f"\n步骤 {step + 1}/{num_steps}")

            # 2. 简单的"决策"（这里只是示例，实际应该使用模型推理）
            # 模拟根据观察生成动作
            controller_state = observation["controllers"].get("left_arm", {})
            current_joint_positions = controller_state.get("qpos", [0.0] * 7)

            # 简单的动作：每个关节增加0.1
            action = [float(pos) + 0.1 for pos in current_joint_positions]

            move_data = {
                "left_arm": {"action_type": "joint", "action": action}
            }

            # 3. 执行动作
            robot.move(move_data)
            print(f"执行动作: {[round(a, 2) for a in action[:3]]}...")

            # 短暂延迟
            time.sleep(0.1)

        # 重置到初始状态
        robot.reset()
        print("\n控制循环完成，机器人已重置")


def example_config_based_initialization():
    """基于配置的初始化示例。"""
    print("\n" + "=" * 50)
    print("示例4: 基于配置的初始化")
    print("=" * 50)

    # 注意：这个示例需要确保MockController和MockCamera已注册
    # 由于MockRobot已经手动创建了控制器和传感器，这里展示配置方式的概念
    print("配置方式初始化（需要确保控制器和传感器已注册）:")
    print(
        """
    config = {
        "controllers": {
            "left_arm": {
                "controller_type": "mock_controller",
                "args": {"name": "left_arm", "num_joints": 7}
            }
        },
        "sensors": {
            "head_camera": {
                "sensor_type": "mock_camera",
                "args": {"name": "head_camera"}
            }
        }
    }
    # robot = MyRobot(config=config)
    """
    )


if __name__ == "__main__":
    """运行所有示例。"""
    try:
        example_basic_usage()
        example_multi_arm_robot()
        example_control_loop()
        example_config_based_initialization()

        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        print("=" * 50)
    except Exception as e:
        print(f"\n示例运行出错: {e}")
        import traceback

        traceback.print_exc()
