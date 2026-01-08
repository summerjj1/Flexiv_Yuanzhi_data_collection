import os
import cv2
import numpy as np
import time
import subprocess
import argparse
from typing import Tuple, List, Optional
from dataclasses import dataclass, field


@dataclass
class IMX415CameraConfig:
    real_time_view: bool = False
    image_size: Tuple[int, int] = (3840, 2160)  # 4K
    fps: int = 30  # 强制30fps
    device_paths: Optional[List[str]] = field(default=None)
    fourcc: str = 'MJPG'
    buffer_size: int = 4


def force_4k_30fps(device_path):
    """强制设置4K 30fps模式"""
    print("🚀 强制启用4K@30fps模式...")
    
    try:
        # 方法1: 使用v4l2-ctl直接设置
        print("1. 设置分辨率和格式...")
        result1 = subprocess.run([
            'v4l2-ctl', '-d', device_path,
            '--set-fmt-video=width=3840,height=2160,pixelformat=MJPG'
        ], capture_output=True, text=True)
        
        if result1.returncode != 0:
            print(f"❌ 格式设置失败: {result1.stderr}")
            return False

        # 方法2: 强制设置30fps
        print("2. 强制设置30fps...")
        result2 = subprocess.run([
            'v4l2-ctl', '-d', device_path, '--set-parm=30'
        ], capture_output=True, text=True)
        
        if result2.returncode != 0:
            print(f"❌ 帧率设置失败: {result2.stderr}")
            return False

        # 方法3: 检查实际设置
        print("3. 验证设置...")
        result3 = subprocess.run([
            'v4l2-ctl', '-d', device_path, '--get-fmt-video'
        ], capture_output=True, text=True)
        
        result4 = subprocess.run([
            'v4l2-ctl', '-d', device_path, '--get-parm'
        ], capture_output=True, text=True)
        
        print("✅ V4L2设置输出:")
        print(result3.stdout)
        print(result4.stdout)
        
        return True
        
    except Exception as e:
        print(f"❌ 设置异常: {e}")
        return False


class HighFPSIMX415:
    def __init__(self, config: IMX415CameraConfig = None):
        if config is None:
            config = IMX415CameraConfig()
        self.config = config
        self.capture = None
        
        if config.device_paths is None:
            self.config.device_paths = ['/dev/video12']
        
        print(f"🎯 目标: 4K@{config.fps}fps")
        self._setup_high_fps_camera()
        
        if config.real_time_view:
            cv2.namedWindow("4K@30fps", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("4K@30fps", 960, 540)

    def _setup_high_fps_camera(self):
        """设置高帧率摄像头"""
        device_path = self.config.device_paths[0]
        
        # 第一步: 强制V4L2设置
        if not force_4k_30fps(device_path):
            print("⚠️  V4L2设置不理想，继续尝试...")
        
        # 第二步: 使用OpenCV的特定后端
        print("4. 使用优化后端打开摄像头...")
        
        # 尝试不同的后端
        backends = [
            cv2.CAP_V4L2,      # V4L2后端
            cv2.CAP_ANY        # 自动选择
        ]
        
        for backend in backends:
            print(f"   尝试后端: {backend}")
            self.capture = cv2.VideoCapture(device_path, backend)
            
            if self.capture.isOpened():
                print(f"   ✅ 后端 {backend} 成功")
                break
            else:
                print(f"   ❌ 后端 {backend} 失败")
        
        if not self.capture or not self.capture.isOpened():
            raise RuntimeError("所有后端都无法打开摄像头")
        
        # 第三步: 精确设置参数
        print("5. 精确设置参数...")
        
        # 先设置格式和分辨率
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 4)
        
        # 设置优化参数
        self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 手动曝光
        self.capture.set(cv2.CAP_PROP_EXPOSURE, 100)
        
        # 第四步: 验证并显示实际设置
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        actual_fourcc = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        
        print(f"📊 最终设置验证:")
        print(f"   分辨率: {actual_width}x{actual_height}")
        print(f"   帧率: {actual_fps}")
        print(f"   格式: {actual_fourcc:#x} ('MJPG' = {cv2.VideoWriter_fourcc(*'MJPG'):#x})")
        
        # 检查是否匹配
        if actual_width != 3840 or actual_height != 2160:
            print("⚠️  警告: 分辨率不匹配!")
        if actual_fps < 25:
            print("⚠️  警告: 帧率可能未达到30fps!")

    def get_frame(self):
        """获取帧（优化版本）"""
        if not self.capture:
            return None
        
        # 使用grab() + retrieve() 提高性能
        grabbed = self.capture.grab()
        if not grabbed:
            return None
        
        ret, frame = self.capture.retrieve()
        return frame if ret else None

    def performance_test(self, duration=10):
        """性能测试"""
        print(f"\n🚀 开始 {duration}秒性能测试...")
        
        frame_count = 0
        start_time = time.time()
        last_display = start_time
        
        # 预热
        for _ in range(5):
            self.get_frame()
        
        print("⏱️  实时性能:")
        print("=" * 50)
        
        try:
            while time.time() - start_time < duration:
                frame = self.get_frame()
                if frame is not None:
                    frame_count += 1
                
                # 每秒显示性能
                current_time = time.time()
                if current_time - last_display >= 1.0:
                    elapsed = current_time - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"  帧率: {current_fps:6.1f} Hz | 总帧数: {frame_count:4d}")
                    last_display = current_time
                
                # 短暂休眠避免过度占用CPU
                time.sleep(0.001)
                    
        except KeyboardInterrupt:
            print("\n⏹️  用户中断测试")
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print("=" * 50)
        print(f"📈 最终结果:")
        print(f"   总时间: {total_time:.1f}s")
        print(f"   总帧数: {frame_count}")
        print(f"   平均帧率: {avg_fps:.1f} Hz")
        
        return avg_fps

    def cleanup(self):
        """清理资源"""
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()


def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description="强制4K@30fps测试")
    parser.add_argument('--test-time', type=int, default=10, help='测试时长(秒)')
    parser.add_argument('--real-time', action='store_true', help='实时预览')
    
    args = parser.parse_args()
    
    config = IMX415CameraConfig(
        real_time_view=args.real_time,
        image_size=(3840, 2160),
        fps=30,
        device_paths=['/dev/video12'],
        fourcc='MJPG',
        buffer_size=4
    )
    
    print("🔄 初始化高帧率4K摄像头...")
    camera = None
    
    try:
        camera = HighFPSIMX415(config)
        
        # 运行性能测试
        fps = camera.performance_test(args.test_time)
        
        if fps >= 25:
            print(f"🎉 成功! 达到 {fps:.1f} Hz (接近30fps)")
        elif fps >= 15:
            print(f"✅ 不错! 达到 {fps:.1f} Hz")
        else:
            print(f"⚠️  帧率较低: {fps:.1f} Hz")
            print("可能原因: USB带宽不足、系统负载高、其他程序占用摄像头")
            
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
    finally:
        if camera:
            camera.cleanup()


if __name__ == '__main__':
    main()