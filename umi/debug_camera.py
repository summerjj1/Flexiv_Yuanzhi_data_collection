import pyrealsense2 as rs
import numpy as np
import cv2

def show_dual_realsense():
    # 1. 获取设备上下文，查询所有连接的设备
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) < 2:
        print(f"错误：需要至少2个摄像头，当前检测到 {len(devices)} 个。")
        return

    pipelines = []
    
    try:
        # 2. 为每个检测到的设备配置 Pipeline
        for i, dev in enumerate(devices):
            serial_number = dev.get_info(rs.camera_info.serial_number)
            print(f"正在配置相机 {i+1}: 序列号 {serial_number}")
            
            pipe = rs.pipeline()
            config = rs.config()
            
            # 关键步骤：绑定特定的序列号，确保不会开启同一个相机两次
            config.enable_device(serial_number)
            
            # 配置 RGB 流 (根据你的相机型号调整分辨率，如 640x480, 1280x720)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # 启动流
            pipe.start(config)
            pipelines.append(pipe)

        print("所有相机已启动，按 'q' 退出...")

        while True:
            frames_list = []
            
            # 3. 循环获取每个 Pipeline 的帧
            for i, pipe in enumerate(pipelines):
                frames = pipe.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # 转换为 numpy 数组
                color_image = np.asanyarray(color_frame.get_data())
                
                # 显示图像
                window_name = f"Camera {i+1}"
                cv2.imshow(window_name, color_image)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"发生错误: {e}")
        
    finally:
        # 4. 资源释放
        for pipe in pipelines:
            pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    show_dual_realsense()