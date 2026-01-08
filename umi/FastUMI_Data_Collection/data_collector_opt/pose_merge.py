from scipy.spatial.transform import Rotation as R
import numpy as np
import os

def read_file_line_by_line(file_path):

    traj_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            traj_list.append(line.strip())

    return traj_list[1:]

def string_to_qpos(traj):
    qpos_list = []
    for line in traj:
        elements = line.split(" ")
        
        if len(elements) != 8:
            print(f"Warning: Line does not have 8 elements: {line}")
        try:
            qpos = [float(elem) for elem in elements]
            qpos_list.append(qpos)
        except ValueError as e:
            print(f"Error converting line to floats: {line} | Error: {e}")
    return qpos_list

def qpos2mat(qpos):
    x, y, z, qx, qy, qz, qw = qpos
    R_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
    t = np.array([x, y, z])
    M = np.eye(4)
    M[:3, :3] = R_mat
    M[:3, 3] = t
    return M

def mat2qpos(M):
    t = M[:3, 3]
    R_mat = M[:3, :3]
    qx, qy, qz, qw = R.from_matrix(R_mat).as_quat()
    return [t[0], t[1], t[2], qx, qy, qz, qw]


TRANSFORMATION = np.eye(4)
TRANSFORMATION[:3, :3] = R.from_euler('xyz', [30, 0, 0], degrees=True).as_matrix()## VIVE iv VIVE flat

def VIVE2VIVE_FLAT(qpos, TRANSFORMATION):
    """
    qpos : x y z qx qy qz qw
    TRANSFORMATION: VIVE iv VIVE flat
    """
    M = qpos2mat(qpos)
    M_flat = np.dot(np.dot(TRANSFORMATION, M), np.linalg.inv(TRANSFORMATION))
    return mat2qpos(M_flat)


def VIVEFLAT2XV(qpos):
    
    TRANSFORMATION_MAT = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    cur_qpos = np.array(qpos)
    cur_mat = qpos2mat(cur_qpos)
    #############################
    left_right_offset = 0.02220
    front_back_offset = -0.04020
    up_down_offset = -0.01003
    #############################
    cur_qpos[0] -= left_right_offset
    cur_qpos[1] -= front_back_offset
    cur_qpos[2] += up_down_offset
    
    ori = cur_mat[:3, :3]
    cur_qpos[:3] += ori[:, 0] * left_right_offset
    cur_qpos[:3] += ori[:, 1] * front_back_offset
    cur_qpos[:3] -= ori[:, 2] * up_down_offset
    
    cur_mat = qpos2mat(cur_qpos)
    
    vive_mat_in_xv =  np.dot(np.dot(TRANSFORMATION_MAT, cur_mat), np.linalg.inv(TRANSFORMATION_MAT)) 
    return mat2qpos(vive_mat_in_xv)


def XV2Gripper(qpos):
    TRANSFORMATION_MAT = np.array([
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    cur_qpos = np.array(qpos)
    cur_mat = qpos2mat(cur_qpos)
    
    #####################################
    left_right_offset = 0.02268
    front_back_offset = 0.08745
    up_down_offset = 0.09240
    #####################################
    cur_qpos[0] -= left_right_offset
    cur_qpos[1] -= up_down_offset
    cur_qpos[2] -= front_back_offset
    
    ori = cur_mat[:3, :3]
    cur_qpos[:3] += ori[:, 0] * left_right_offset
    cur_qpos[:3] += ori[:, 1] * up_down_offset
    cur_qpos[:3] += ori[:, 2] * front_back_offset
    
    cur_mat = qpos2mat(cur_qpos)
    xv_mat_in_gripper =  np.dot(np.dot(TRANSFORMATION_MAT, cur_mat), np.linalg.inv(TRANSFORMATION_MAT))
    return mat2qpos(xv_mat_in_gripper)


def transform_vive_to_gripper(qpos):
    TRANSFORMATION = np.eye(4)
    TRANSFORMATION[:3, :3] = R.from_euler('xyz', [30, 0, 0], degrees=True).as_matrix()
    
    # VIVE → VIVE_FLAT
    qpos = VIVE2VIVE_FLAT(qpos, TRANSFORMATION)
    
    # VIVE_FLAT → XV
    qpos = VIVEFLAT2XV(qpos)
    
    # XV → Gripper
    qpos = XV2Gripper(qpos)
    
    return qpos


def transform_slam_to_gripper(qpos):
    # XV → Gripper
    qpos = XV2Gripper(qpos)
    
    return qpos
    
    

if __name__ == "__main__":

    TRANSFORMATION_MAT = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    

    session_name = "session_002"
    session_path = f"/home/onestar/jiawei/data_collection/data_collector/250801DR48FP25002993_multi_sessions/{session_name}"
    vive_path = f"{session_path}/Vive_Poses_Offset/vive_data_tum.txt"
    slam_path = f"{session_path}/SLAM_Poses/slam_raw.txt"
    vive_target_path = f"{session_path}/Vive_Poses_in_SLAM_Coordinate/vive_poses_in_slam_coordinate.txt"

    print(f"读取VIVE数据: {vive_path}")
    print(f"读取SLAM数据: {slam_path}")
    print(f"输出转换后的VIVE数据到: {vive_target_path}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(vive_target_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    vive_traj = read_file_line_by_line(vive_path)
    slam_traj = read_file_line_by_line(slam_path)
    
    vive_qpos = string_to_qpos(vive_traj)
    slam_qpos = string_to_qpos(slam_traj)
    
    vive_qpos_in_xv = []
    vive_qpos_in_gripper = []
    for vive_item in vive_qpos:
        cur_qpos = np.array(vive_item[1:])
        cur_qpos = VIVE2VIVE_FLAT(cur_qpos, TRANSFORMATION)

        cur_pose_in_xv = VIVEFLAT2XV(cur_qpos)
        vive_qpos_in_xv.append([vive_item[0]]+cur_pose_in_xv)
    
    with open(vive_target_path, 'w') as f:
        f.write("# timestamp x y z qx qy qz qw\n")
        for data in vive_qpos_in_xv:
            f.write(f"{data[0]} {data[1]} {data[2]} {data[3]} {data[4]} {data[5]} {data[6]} {data[7]}\n")
    
