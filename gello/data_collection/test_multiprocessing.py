import multiprocessing
import pyrealsense2 as rs
import time
import numpy as np
from datetime import datetime

def camera_process(camera_id, serial_num, log_queue):
    """
    Process to capture frames from a Realsense camera and log timestamps.
    Args:
        camera_id (int): Identifier for the camera (e.g., 1 or 2).
        serial_num (str): Serial number of the Realsense camera.
        log_queue (multiprocessing.Queue): Queue to store timestamp logs.
    """
    try:
        # Initialize pipeline for the specific camera
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_num)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        pipeline.start(config)
        print(f"Camera {camera_id} (serial: {serial_num}) started")

        frame_count = 0
        start_time = time.perf_counter()
        
        while frame_count < 100:  # Capture 100 frames for testing
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
                
            frame_count += 1
            timestamp = time.perf_counter() - start_time
            log_queue.put({
                'camera_id': camera_id,
                'frame': frame_count,
                'timestamp': timestamp,
                'datetime': datetime.now().strftime('%H:%M:%S.%f')
            })
            
            # Optional: Basic frame processing to simulate workload
            color_data = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())
            
            # Simulate some processing delay
            time.sleep(0.01)
            
        pipeline.stop()
        log_queue.put({
            'camera_id': camera_id,
            'frame': -1,  # Signal process end
            'timestamp': time.perf_counter() - start_time,
            'datetime': datetime.now().strftime('%H:%M:%S.%f')
        })
        print(f"Camera {camera_id} (serial: {serial_num}) stopped")
        
    except Exception as e:
        log_queue.put({
            'camera_id': camera_id,
            'frame': -1,
            'timestamp': -1,
            'datetime': datetime.now().strftime('%H:%M:%S.%f'),
            'error': str(e)
        })

def main():
    # Initialize context to get connected Realsense devices
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) < 2:
        print(f"Error: Only {len(devices)} device(s) found. Need 2 Realsense cameras.")
        return

    # Get serial numbers of the two cameras
    serials = [device.get_info(rs.camera_info.serial_number) for device in devices]
    print(f"Found cameras with serials: {serials}")

    # Create a queue for logging timestamps
    log_queue = multiprocessing.Queue()
    
    # Start processes for each camera
    processes = [
        multiprocessing.Process(target=camera_process, args=(1, serials[0], log_queue)),
        multiprocessing.Process(target=camera_process, args=(2, serials[1], log_queue))
    ]
    
    for p in processes:
        p.start()
    
    # Collect and print logs
    finished = 0
    logs = {1: [], 2: []}
    
    while finished < 2:
        log = log_queue.get()
        camera_id = log['camera_id']
        logs[camera_id].append(log)
        
        if 'error' in log:
            print(f"Error in camera {camera_id}: {log['error']}")
            finished += 1
        elif log['frame'] == -1:
            finished += 1
        
        print(f"Camera {camera_id} | Frame {log['frame']} | Time {log['timestamp']:.3f}s | {log['datetime']}")
    
    # Analyze logs to verify parallelism
    for camera_id in logs:
        timestamps = [log['timestamp'] for log in logs[camera_id] if log['frame'] != -1]
        if timestamps:
            avg_interval = np.mean(np.diff(timestamps))
            print(f"Camera {camera_id} average frame interval: {avg_interval:.3f}s (~{1/avg_interval:.1f} FPS)")

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()