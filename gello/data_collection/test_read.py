import pyrealsense2 as rs
import time
import numpy as np
from datetime import datetime
from typing import List, Tuple

class SequentialCamera:
    def __init__(self, serials: List[str]):
        self.serials = serials
        self.pipelines = []
        self.profiles = []
        self.align = rs.align(rs.stream.color)
        
        # Initialize pipelines
        for serial in serials:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            try:
                pipeline.start(config)
                self.pipelines.append(pipeline)
                self.profiles.append(pipeline.get_active_profile())
                print(f"Camera with serial {serial} started")
            except Exception as e:
                print(f"Failed to start camera with serial {serial}: {e}")
                raise
        self.logs = {i: [] for i in range(len(serials))}
        self.start_time = time.perf_counter()
    
    def get_camera_intrinsics(self, profile):
        """Get camera intrinsics and distortion coefficients"""
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = stream.get_intrinsics()
        cam_intrinsics = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                  [0, intrinsics.fy, intrinsics.ppy],
                                  [0, 0, 1]])
        dist_coeffs = np.array(intrinsics.coeffs)
        return cam_intrinsics, dist_coeffs
    
    def get_data(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]]:
        """Get data from all cameras sequentially"""
        framesets = []
        data = []
        call_start = time.perf_counter()
        
        try:
            # Sequentially wait for frames from each camera
            for i, pipeline in enumerate(self.pipelines):
                frames = pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                framesets.append(aligned_frames)
            
            # Process frames
            for i, frames in enumerate(framesets):
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    print(f"Camera {i} failed to get valid frames")
                    self.logs[i].append({
                        'camera_id': i,
                        'frame': -1,
                        'timestamp': time.perf_counter() - self.start_time,
                        'datetime': datetime.now().strftime('%H:%M:%S.%f'),
                        'error': 'Invalid frame'
                    })
                    return None
                
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                cam_intrinsics, dist_coeffs = self.get_camera_intrinsics(self.profiles[i])
                
                # Log timestamp
                timestamp = time.perf_counter() - self.start_time
                self.logs[i].append({
                    'camera_id': i,
                    'frame': len(self.logs[i]) + 1,
                    'timestamp': timestamp,
                    'datetime': datetime.now().strftime('%H:%M:%S.%f')
                })
                
                data.append((color_image, depth_image, cam_intrinsics, dist_coeffs))
            
            call_duration = time.perf_counter() - call_start
            print(f"get_data call took {call_duration:.3f}s")
            return data
        
        except RuntimeError as e:
            print(f"Error capturing frames: {e}")
            self.logs[i].append({
                'camera_id': i,
                'frame': -1,
                'timestamp': time.perf_counter() - self.start_time,
                'datetime': datetime.now().strftime('%H:%M:%S.%f'),
                'error': str(e)
            })
            return None
    
    def stop(self):
        """Stop all pipelines"""
        for pipeline in self.pipelines:
            pipeline.stop()
    
    def print_logs(self):
        """Print logs and analyze frame intervals"""
        for camera_id in self.logs:
            timestamps = [log['timestamp'] for log in self.logs[camera_id] if 'error' not in log]
            if timestamps:
                avg_interval = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0
                print(f"Camera {camera_id} logs:")
                for log in self.logs[camera_id]:
                    if 'error' in log:
                        print(f"Camera {log['camera_id']} | Error: {log['error']} | "
                              f"Time {log['timestamp']:.3f}s | {log['datetime']}")
                    else:
                        print(f"Camera {log['camera_id']} | Frame {log['frame']} | "
                              f"Time {log['timestamp']:.3f}s | {log['datetime']}")
                print(f"Camera {camera_id} average frame interval: {avg_interval:.3f}s "
                      f"(~{1/avg_interval:.1f} FPS)" if avg_interval else "No interval data")

def main():
    # Initialize context to get connected Realsense devices
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) < 2:
        print(f"Error: Only {len(devices)} device(s) found. Need 2 Realsense cameras.")
        return
    
    serials = [device.get_info(rs.camera_info.serial_number) for device in devices]
    print(f"Found cameras with serials: {serials}")
    
    cameras = SequentialCamera(serials)
    total_start = time.perf_counter()
    
    try:
        # Capture 100 sets of data
        for _ in range(100):
            data = cameras.get_data()
            if data:
                for i, (color, depth, intr, dist) in enumerate(data):
                    print(f"Camera {i}: Color shape {color.shape}, Depth shape {depth.shape}")
            else:
                print("No data returned, skipping...")
        
    except Exception as e:
        print(f"Main loop error: {e}")
    finally:
        cameras.stop()
        total_duration = time.perf_counter() - total_start
        print(f"Total execution time: {total_duration:.3f}s")
        cameras.print_logs()

if __name__ == '__main__':
    main()