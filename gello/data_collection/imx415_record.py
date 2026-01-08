import os
import cv2
import numpy as np
import time
import subprocess
import argparse
import fcntl
import mmap
import select
import struct
from typing import Tuple, List, Optional
from dataclasses import dataclass, field

# V4L2 constants
VIDIOC_QUERYCAP = 0x80685600
VIDIOC_ENUM_FMT = 0xc0405602
VIDIOC_S_FMT = 0xc0d05605
VIDIOC_G_FMT = 0xc0d05604
VIDIOC_REQBUFS = 0xc0145608
VIDIOC_QUERYBUF = 0xc0445609
VIDIOC_QBUF = 0xc044560f
VIDIOC_DQBUF = 0xc0445611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_ANY = 0

# 像素格式
V4L2_PIX_FMT_YUYV = 0x56595559  # YUYV 4:2:2
V4L2_PIX_FMT_MJPEG = 0x47504A4D  # MJPG

@dataclass
class V4L2CameraConfig:
    real_time_view: bool = False
    image_size: Tuple[int, int] = (3840, 2160)  # 4K
    fps: int = 60  # 强制30fps
    device_path: Optional[List[str]] = field(default=None)
    fourcc: str = 'MJPG'
    buffer_size: int = 4
    device_paths: Optional[List[str]] = field(default=None)
    pixel_format: str = 'MJPG'  # 'YUYV' or 'MJPG'


class V4L2Camera:
    """基于V4L2的高性能摄像头类"""
    def __init__(self, config: V4L2CameraConfig = None):

    # def __init__(self, device_path: str, width: int = 1280, height: int = 720, 
    #              pixel_format: str = 'YUYV', fps: int = 30, buffer_count: int = 4):
        self.config = config
        self.device_path = config.device_path[0] if config.device_path else '/dev/video12'
        self.width = config.image_size[0]
        self.height = config.image_size[1]
        self.fps = config.fps
        self.buffer_count = config.buffer_size
        self.fd = None
        self.buffers = []
        self.streaming = False
        self.device_count = 1
        # 设置像素格式
        if self.config.pixel_format == 'MJPEG':
            self.pixel_format = V4L2_PIX_FMT_MJPEG
        else:
            self.pixel_format = V4L2_PIX_FMT_YUYV
            
        self.pixel_format_name = self.config.pixel_format
        
        self._open_device()
        self._setup_format()
        self._setup_buffers()
        
    def _open_device(self):
        """打开V4L2设备"""
        try:
            self.fd = os.open(self.device_path, os.O_RDWR | os.O_NONBLOCK)
            print(f"✅ V4L2设备已打开: {self.device_path}")
        except OSError as e:
            raise RuntimeError(f"无法打开V4L2设备 {self.device_path}: {e}")
    
    def _setup_format(self):
        """设置视频格式"""
        # 构造格式结构体
        fmt = struct.pack(
            '=LLLLLLLLLLLLLL',
            V4L2_BUF_TYPE_VIDEO_CAPTURE,  # type
            self.width,                    # width
            self.height,                   # height
            self.pixel_format,             # pixelformat
            V4L2_FIELD_ANY,               # field
            self.width * 2 if self.pixel_format == V4L2_PIX_FMT_YUYV else 0,  # bytesperline
            self.width * self.height * 2 if self.pixel_format == V4L2_PIX_FMT_YUYV else 0,  # sizeimage
            0, 0, 0, 0, 0, 0, 0           # 剩余字段
        )
        
        try:
            fcntl.ioctl(self.fd, VIDIOC_S_FMT, fmt)
            print(f"✅ 设置格式: {self.width}x{self.height} {self.pixel_format_name}")
        except OSError as e:
            raise RuntimeError(f"设置视频格式失败: {e}")
    
    def _setup_buffers(self):
        """设置缓冲区"""
        # 请求缓冲区
        reqbuf = struct.pack('=LLLL', self.buffer_count, V4L2_BUF_TYPE_VIDEO_CAPTURE, 
                            V4L2_MEMORY_MMAP, 0)
        
        try:
            fcntl.ioctl(self.fd, VIDIOC_REQBUFS, reqbuf)
            print(f"✅ 请求了 {self.buffer_count} 个缓冲区")
        except OSError as e:
            raise RuntimeError(f"请求缓冲区失败: {e}")
        
        # 查询并映射每个缓冲区
        for i in range(self.buffer_count):
            querybuf = struct.pack('=LLLLLLLLLLLL', i, V4L2_BUF_TYPE_VIDEO_CAPTURE,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            try:
                result = fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, querybuf)
                # 解析返回的缓冲区信息
                unpacked = struct.unpack('=LLLLLLLLLLLL', result)
                offset = unpacked[9]  # m.offset
                length = unpacked[4]  # length
                
                # 内存映射
                buffer_mem = mmap.mmap(self.fd, length, 
                                     mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                                     offset=offset)
                
                self.buffers.append({
                    'index': i,
                    'length': length,
                    'offset': offset,
                    'mmap': buffer_mem
                })
                
            except OSError as e:
                raise RuntimeError(f"查询/映射缓冲区 {i} 失败: {e}")
        
        print(f"✅ 映射了 {len(self.buffers)} 个缓冲区")
    
    def start_streaming(self):
        """开始流传输"""
        if self.streaming:
            return
            
        # 将所有缓冲区加入队列
        for buffer_info in self.buffers:
            qbuf = struct.pack('=LLLLLLLLLLLL', buffer_info['index'], 
                              V4L2_BUF_TYPE_VIDEO_CAPTURE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            try:
                fcntl.ioctl(self.fd, VIDIOC_QBUF, qbuf)
            except OSError as e:
                raise RuntimeError(f"缓冲区入队失败: {e}")
        
        # 启动流
        buf_type = struct.pack('=L', V4L2_BUF_TYPE_VIDEO_CAPTURE)
        try:
            fcntl.ioctl(self.fd, VIDIOC_STREAMON, buf_type)
            self.streaming = True
            print("✅ 流传输已启动")
        except OSError as e:
            raise RuntimeError(f"启动流失败: {e}")
    
    def stop_streaming(self):
        """停止流传输"""
        if not self.streaming:
            return
            
        buf_type = struct.pack('=L', V4L2_BUF_TYPE_VIDEO_CAPTURE)
        try:
            fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, buf_type)
            self.streaming = False
            print("✅ 流传输已停止")
        except OSError as e:
            print(f"停止流失败: {e}")
    
    def read_frame(self, timeout=1.0):
        """读取一帧数据"""
        if not self.streaming:
            self.start_streaming()
        
        # 使用select等待数据就绪
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return None
        
        # 出队一个缓冲区
        dqbuf = struct.pack('=LLLLLLLLLLLL', 0, V4L2_BUF_TYPE_VIDEO_CAPTURE,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        try:
            result = fcntl.ioctl(self.fd, VIDIOC_DQBUF, dqbuf)
            unpacked = struct.unpack('=LLLLLLLLLLLL', result)
            index = unpacked[0]
            bytesused = unpacked[4]
            
            # 读取数据
            buffer_info = self.buffers[index]
            buffer_info['mmap'].seek(0)
            data = buffer_info['mmap'].read(bytesused)
            
            # 重新入队
            qbuf = struct.pack('=LLLLLLLLLLLL', index, V4L2_BUF_TYPE_VIDEO_CAPTURE,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            fcntl.ioctl(self.fd, VIDIOC_QBUF, qbuf)
            
            # 转换数据为numpy数组
            if self.pixel_format == V4L2_PIX_FMT_MJPEG:
                # MJPEG需要解码
                frame = self._decode_mjpeg(data)
            else:
                # YUYV转BGR
                frame = self._yuyv_to_bgr(data)
            
            return frame
            
        except OSError as e:
            print(f"读取帧失败: {e}")
            return None
    
    def _decode_mjpeg(self, data):
        """解码MJPEG数据"""
        try:
            # 使用OpenCV解码MJPEG
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"MJPEG解码失败: {e}")
            return None
    
    def _yuyv_to_bgr(self, data):
        """将YUYV格式转换为BGR"""
        try:
            # YUYV是4:2:2格式，每2个像素占4字节
            yuyv = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width * 2))
            
            # 转换为BGR
            bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
            return bgr
        except Exception as e:
            print(f"YUYV转换失败: {e}")
            return None
    
    def cleanup(self):
        """清理资源"""
        self.stop_streaming()
        
        # 解除内存映射
        for buffer_info in self.buffers:
            if buffer_info['mmap']:
                buffer_info['mmap'].close()
        
        # 关闭设备
        if self.fd is not None:
            os.close(self.fd)
            print("✅ V4L2设备已关闭")




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
            'v4l2-ctl', '-d', device_path, '--set-parm=60'
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


# class IMX415Module:
#     def __init__(self, config: IMX415CameraConfig = None):
#         if config is None:
#             config = IMX415CameraConfig()
#         self.config = config
#         self.cameras = []  # 改为存储V4L2Camera对象
#         self.device_count = 0
        
#         if config.device_paths is None:
#             self.config.device_paths = ['/dev/video12']
        
#         print(f"🎯 目标: 4K@{config.fps}fps")
#         # self._setup_high_fps_camera()
        
#         if config.real_time_view:
#             for i in range(len(self.config.device_paths)):
#                 device_name = os.path.basename(self.config.device_paths[i])
#                 cv2.namedWindow(f"IMX415_{device_name}", cv2.WINDOW_AUTOSIZE)
    
    
#     def _setup_cameras(self):
#         """设置所有V4L2摄像头"""
#         for device_path in self.config.device_paths:
#             print(f"初始化V4L2摄像头: {device_path}")
            
#             try:
#                 # 创建V4L2Camera实例
#                 camera = V4L2Camera(
#                     device_path=device_path,
#                     width=self.config.image_size[0],
#                     height=self.config.image_size[1],
#                     pixel_format=self.config.fourcc,
#                     fps=self.config.fps,
#                     buffer_count=self.config.buffer_size
#                 )
                
#                 # 测试读取一帧
#                 test_frame = camera.read_frame(timeout=2.0)
#                 if test_frame is not None:
#                     print(f"✅ V4L2摄像头初始化成功: {device_path}")
#                     print(f"   实际分辨率: {test_frame.shape[1]}x{test_frame.shape[0]}")
#                     print(f"   像素格式: {self.config.fourcc}")
#                     self.cameras.append(camera)
#                 else:
#                     print(f"❌ 无法从V4L2设备读取帧: {device_path}")
#                     camera.cleanup()
                    
#             except Exception as e:
#                 print(f"❌ V4L2摄像头初始化失败 {device_path}: {e}")
#                 continue
        
#         self.device_count = len(self.cameras)
#         if self.device_count == 0:
#             error_msg = f"无法初始化任何V4L2摄像头设备: {self.config.device_paths}\n"
#             error_msg += "请确认:\n"
#             error_msg += "1. 设备路径是否正确\n"
#             error_msg += "2. 摄像头权限是否足够: sudo chmod 666 /dev/video*\n" 
#             error_msg += "3. 设备是否被其他程序占用\n"
#             error_msg += "4. 设备是否支持指定的分辨率和格式"
#             raise RuntimeError(error_msg)
    
#     def get_camera_info(self, cap_index: int = 0) -> dict:
#         """获取摄像头信息"""
#         if cap_index >= len(self.cameras):
#             return {}
        
#         camera = self.cameras[cap_index]
#         info = {
#             'width': camera.width,
#             'height': camera.height,
#             'fps': camera.fps,
#             'pixel_format': camera.pixel_format_name,
#             'buffer_count': camera.buffer_count,
#             'device_path': camera.device_path,
#             'streaming': camera.streaming
#         }
#         return info
    
#     def get_data(self) -> List[np.ndarray]:
#         """从所有摄像头获取图像数据"""
#         images = []
        
#         for i, camera in enumerate(self.cameras):
#             frame = camera.read_frame(timeout=0.1)  # 100ms超时
#             if frame is None:
#                 print(f"警告: 无法从V4L2摄像头 {self.config.device_paths[i]} 读取帧")
#                 # 返回空图像作为占位符
#                 empty_frame = np.zeros((self.config.image_size[1], self.config.image_size[0], 3), dtype=np.uint8)
#                 images.append(empty_frame)
#             else:
#                 print(f"   ❌ 后端 {backend} 失败")
        
#         return images
    
#     def get_single_frame(self, cam_index: int = 0) -> Optional[np.ndarray]:
#         """从指定摄像头获取单帧图像"""
#         if cam_index >= len(self.cameras):
#             print(f"错误: 摄像头索引 {cam_index} 超出范围")
#             return None
        
#         return self.cameras[cam_index].read_frame(timeout=1.0)
    
#     def show_real_time_view(self):
#         """显示实时预览"""
#         if not self.config.real_time_view:
#             return
        
#         ret, frame = self.capture.retrieve()
#         return frame if ret else None

#     def performance_test(self, duration=10):
#         """性能测试"""
#         print(f"\n🚀 开始 {duration}秒性能测试...")
        
#         frame_count = 0
#         start_time = time.time()
#         last_display = start_time
        
#         # 预热
#         for _ in range(5):
#             self.get_frame()
        
#         print("⏱️  实时性能:")
#         print("=" * 50)
        
#         try:
#             while time.time() - start_time < duration:
#                 frame = self.get_frame()
#                 if frame is not None:
#                     frame_count += 1
                
#                 # 每秒显示性能
#                 current_time = time.time()
#                 if current_time - last_display >= 1.0:
#                     elapsed = current_time - start_time
#                     current_fps = frame_count / elapsed if elapsed > 0 else 0
#                     print(f"  帧率: {current_fps:6.1f} Hz | 总帧数: {frame_count:4d}")
#                     last_display = current_time
                
#                 # 短暂休眠避免过度占用CPU
#                 time.sleep(0.001)
                    
#         except KeyboardInterrupt:
#             print("\n⏹️  用户中断测试")
        
#         total_time = time.time() - start_time
#         avg_fps = frame_count / total_time if total_time > 0 else 0
        
#         print("=" * 50)
#         print(f"📈 最终结果:")
#         print(f"   总时间: {total_time:.1f}s")
#         print(f"   总帧数: {frame_count}")
#         print(f"   平均帧率: {avg_fps:.1f} Hz")
        
#         return avg_fps

#     def cleanup(self):
#         """清理资源"""
#         for camera in self.cameras:
#             try:
#                 camera.cleanup()
#             except Exception as e:
#                 print(f"清理V4L2摄像头时出错: {e}")
        
#         if self.config.real_time_view:
#             try:
#                 cv2.destroyAllWindows()
#             except Exception as e:
#                 print(f"关闭窗口时出错: {e}")


def get_images(imx415_module: V4L2Camera) -> List[np.ndarray]:
    """获取所有摄像头的图像"""
    # import pdb; pdb.set_trace()
    return [imx415_module.get_frame()]


def parse_args() -> V4L2CameraConfig:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="IMX415 Camera Data Capture")
    parser.add_argument('--real-time-view', action='store_true', help='启用实时预览')
    parser.add_argument('--width', type=int, default=1920, help='图像宽度')
    parser.add_argument('--height', type=int, default=1080, help='图像高度')
    parser.add_argument('--fps', type=int, default=30, help='帧率')
    parser.add_argument('--save-path', type=str, default='./imx415/images', 
                       help='图像保存路径')
    parser.add_argument('--save-freq', type=int, default=30, help='保存频率 (Hz)')
    # parser.add_argument('--device-path', type=str, nargs='+', 
    #                    help='摄像头设备路径列表，如 /dev/video0 /dev/video1')
    parser.add_argument('--fourcc', type=str, default='YUYV', 
                       help='像素格式 (YUYV, MJPEG)')
    parser.add_argument('--buffer-size', type=int, default=1, help='缓冲区大小')
    
    args = parser.parse_args()
    
    config = V4L2CameraConfig(
        real_time_view=args.real_time,
        image_size=(3840, 2160),
        fps=60,
        device_paths=['/dev/video12'],
        fourcc='MJPG',
        buffer_size=4
    )
    
    print("🔄 初始化高帧率4K摄像头...")
    camera = None
    
    try:
        camera = V4L2Camera(config)
        
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
        try:
            camera.cleanup()
        except:
            pass
        cv2.destroyAllWindows()

def test_v4l2_performance():
    """测试V4L2不同配置的性能"""
    test_configs = [
        {'size': (640, 480), 'format': 'YUYV', 'buffers': 4},
        {'size': (640, 480), 'format': 'MJPEG', 'buffers': 4},
        {'size': (1280, 720), 'format': 'YUYV', 'buffers': 4},
        {'size': (1280, 720), 'format': 'MJPEG', 'buffers': 4},
        {'size': (1920, 1080), 'format': 'YUYV', 'buffers': 4},
        {'size': (1920, 1080), 'format': 'MJPEG', 'buffers': 4},
    ]
    
    print("=== V4L2性能测试 ===")
    
    for config in test_configs:
        print(f"\n📊 测试: {config['size']} {config['format']}")
        
        cam_config = V4L2CameraConfig(
            image_size=config['size'],
            fourcc=config['format'],
            buffer_size=config['buffers'],
            device_paths=['/dev/video12']
        )
        import pdb; pdb.set_trace()
        try:
            camera = V4L2Camera(cam_config)
            
            # 预热
            for _ in range(3):
                camera.get_data()
            
            # 测试50帧的性能
            frame_count = 50
            start_time = time.time()
            
            successful_frames = 0
            for i in range(frame_count):
                images = camera.get_data()
                if images and len(images) > 0 and images[0] is not None:
                    successful_frames += 1
            
            total_time = time.time() - start_time
            avg_fps = successful_frames / total_time
            print(f"   ✅ 实际帧率: {avg_fps:.2f} fps")
            print(f"   📈 成功率: {successful_frames}/{frame_count} ({100*successful_frames/frame_count:.1f}%)")
            print(f"   ⏱️  总时间: {total_time:.2f}s")
            
            camera.cleanup()
            
        except Exception as e:
            print(f"   ❌ 配置失败: {e}")

def compare_opencv_vs_v4l2():
    """比较OpenCV和V4L2的性能"""
    print("\n=== OpenCV vs V4L2 性能比较 ===")
    
    # 测试OpenCV
    print("\n🔵 OpenCV VideoCapture:")
    try:
        cap = cv2.VideoCapture('/dev/video12')
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 预热
            for _ in range(5):
                cap.read()
            
            start_time = time.time()
            successful = 0
            for _ in range(30):
                ret, frame = cap.read()
                if ret:
                    successful += 1
            
            total_time = time.time() - start_time
            fps = successful / total_time
            print(f"   帧率: {fps:.2f} fps, 成功率: {successful}/30")
            cap.release()
        else:
            print("   无法打开OpenCV设备")
    except Exception as e:
        print(f"   OpenCV测试失败: {e}")
    
    # 测试V4L2
    print("\n🟢 V4L2直接访问:")
    try:
        v4l2_cam = V4L2Camera(device_path='/dev/video12', width=1280, height=720, pixel_format='YUYV', fps=30, buffer_count=4)

        # 预热
        for _ in range(5):
            v4l2_cam.read_frame()
        
        start_time = time.time()
        successful = 0
        for _ in range(30):
            frame = v4l2_cam.read_frame(timeout=0.1)
            if frame is not None:
                successful += 1
        
        total_time = time.time() - start_time
        fps = successful / total_time
        print(f"   帧率: {fps:.2f} fps, 成功率: {successful}/30")
        v4l2_cam.cleanup()
    except Exception as e:
        print(f"   V4L2测试失败: {e}")

if __name__ == '__main__':
    # 性能测试
    test_v4l2_performance()
    # compare_opencv_vs_v4l2()
# V4L2高性能IMX415摄像头模块使用示例:

# 1. 性能测试:
# python imx415_record.py  # 运行性能对比测试

# 2. 基本使用 (默认使用 /dev/video12):
# python imx415_record.py --real-time-view

# 3. 指定设备和格式:
# python imx415_record.py --device-paths /dev/video12 --fourcc YUYV --real-time-view
# python imx415_record.py --device-paths /dev/video12 --fourcc MJPEG --real-time-view

# 4. 高分辨率设置:
# python imx415_record.py --device-paths /dev/video12 --width 1920 --height 1080 --fourcc MJPEG --real-time-view

# 5. 多摄像头 (如果有):
# python imx415_record.py --device-paths /dev/video12 /dev/video13 --real-time-view

# 6. 优化设置 (减少延迟):
# python imx415_record.py --device-paths /dev/video12 --fourcc YUYV --buffer-size 2 --real-time-view

# 注意事项:
# - YUYV格式CPU占用更高但延迟更低
# - MJPEG格式CPU占用更低但可能有轻微延迟
# - 较小的buffer_size可以减少延迟
# - 确保设备权限: sudo chmod 666 /dev/video*
