import flexivrdk
import time
from pdb import set_trace

def test_gripper_with_correct_api():
    print("=== 使用正确API测试夹爪 ===")
    
    # 初始化机器人
    robot = flexivrdk.Robot("Rizon 4s-063036")
    robot.Enable()
    
    # 等待就绪
    for i in range(50):
        if robot.operational():
            print("✓ 机器人就绪")
            break
        time.sleep(0.1)
    
    # 初始化夹爪
    gripper = flexivrdk.Gripper(robot)
    gripper.Enable("Flexiv-GN01")
    # gripper.Enable("GripperDahuanModbus")


    time.sleep(2)
    print("✓ 夹爪使能成功")
    
    # 可选：初始化夹爪（如果需要）
    try:
        # gripper.Init()
        print("✓ 夹爪初始化完成")
        time.sleep(1)
    except Exception as e:
        print(f"初始化失败（可能是可选的）: {e}")
    
    
    # 测试2: 使用Move方法 - 尝试不同的参数
    print("\n=== 测试Move方法 ===")
    
    gripper_max_vel = gripper.params().max_vel - 0.01
    # 根据之前错误，力值需要调整。大寰夹爪通常力值范围较小
    test_cases = [
        # (位置, 速度, 力值, 描述)
        (0.0, 0.19, 70, "闭合 - 小力值"),
        (0.1, 0.19, 50, "非常轻微的力"),
    ]
    for pos, speed, force, desc in test_cases:
        print(f"\n{desc}: 位置={pos}, 速度={speed}, 力={force}")
        # set_trace()
        try:
            gripper.Move(pos, 0.19, 50)
            # from pdb import set_trace; set_trace()
            print("  ✓ 命令发送成功")
            
            # 等待执行
            for i in range(40):  # 4秒超时
                if not robot.busy():
                    print("  ✓ 执行完成")
                    break
                time.sleep(0.1)
            else:
                print("  ⚠ 等待超时")
            # 显示当前状态
            try:
                print(f"  当前状态: {gripper.states().width}")
            except:
                pass
                
            time.sleep(1)  # 稳定时间
    

        except Exception as e:
            print(f"  ✗ 失败: {e}")


    print("\n=== 测试完成 ===")
    # gripper.Move(0.1, 0.01, 50)  # 打开夹爪
    # time.sleep(5)

if __name__ == "__main__":
    test_gripper_with_correct_api()