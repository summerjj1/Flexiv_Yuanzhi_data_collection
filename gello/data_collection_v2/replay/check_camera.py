import pyrealsense2 as rs
import cv2
import numpy as np

# 创建管道
pipeline = rs.pipeline()
config = rs.config()

# 配置相机
config.enable_device("335222075526")
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动流
pipeline.start(config)

try:
    while True:
        # 获取帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        # 转换为图像
        color_image = np.asanyarray(color_frame.get_data())
        
        # 显示
        cv2.imshow('D435i RGB', color_image)
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()