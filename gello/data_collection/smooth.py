import pykalman 
import numpy as np

# class KalmanFilterPose:
#     def __init__(self):
#         # 状态: [x, y, z, vx, vy, vz, qw, qx, qy, qz]
#         self.dim_state = 10
#         self.kf = pykalman.KalmanFilter(
#             transition_matrices=np.eye(self.dim_state),
#             observation_matrices=np.eye(self.dim_state),
#             initial_state_mean=np.zeros(self.dim_state),
#             initial_state_covariance=np.eye(self.dim_state) * 0.1,
#             observation_covariance=np.eye(self.dim_state) * 0.1,
#             transition_covariance=np.eye(self.dim_state) * 0.01
#         )
#         self.current_state = None
        
#     def update(self, position, quaternion):
#         observation = np.concatenate([
#             position,
#             np.zeros(3),  # 速度未知
#             [quaternion.w, quaternion.x, quaternion.y, quaternion.z]
#         ])
        
#         if self.current_state is None:
#             self.current_state = self.kf.initial_state_mean
#             self.current_covariance = self.kf.initial_state_covariance
        
#         self.current_state, self.current_covariance = self.kf.filter_update(
#             self.current_state, self.current_covariance, observation
#         )
        
#         smoothed_pos = self.current_state[0:3]
#         smoothed_quat = self.current_state[6:10]
#         smoothed_quat = smoothed_quat / np.linalg.norm(smoothed_quat)  # 归一化
#         cmd_quat_numpy = np.quaternion(smoothed_quat[0], smoothed_quat[1], smoothed_quat[2], smoothed_quat[3])
#         return smoothed_pos, cmd_quat_numpy
    


class KalmanFilterPose:
    def __init__(self):
        # 只滤波位置，不处理四元数
        self.dim_state = 6  # [x, y, z, vx, vy, vz]
        
        # 状态转移矩阵（位置 = 位置 + 速度 * dt）
        dt = 0.01  # 假设10ms更新一次，根据实际调整
        transition = np.eye(self.dim_state)
        transition[0:3, 3:6] = np.eye(3) * dt
        
        self.kf = pykalman.KalmanFilter(
            transition_matrices=transition,
            observation_matrices=np.eye(self.dim_state),
            initial_state_mean=np.zeros(self.dim_state),
            initial_state_covariance=np.eye(self.dim_state) * 0.1,
            observation_covariance=np.eye(self.dim_state) * 0.05,  # 观测噪声
            transition_covariance=np.eye(self.dim_state) * 0.01     # 过程噪声
        )
        self.current_state = None
        self.current_covariance = None
        
    def update(self, position, quaternion):
        """
        更新滤波器
        Args:
            position: np.array [x, y, z]
            quaternion: np.quaternion(w, x, y, z)
        Returns:
            smoothed_pos: 滤波后的位置
            quaternion: 原始四元数（不做处理）
        """
        # 构建观测向量（位置 + 零速度）
        observation = np.concatenate([position, np.zeros(3)])
        
        # 初始化状态
        if self.current_state is None:
            self.current_state = observation.copy()
            self.current_covariance = self.kf.initial_state_covariance
        
        # 卡尔曼滤波更新
        self.current_state, self.current_covariance = self.kf.filter_update(
            self.current_state, 
            self.current_covariance, 
            observation
        )
        
        # 提取滤波后的位置
        smoothed_pos = self.current_state[0:3]
        
        # 四元数直接返回，不做任何处理
        return smoothed_pos, quaternion