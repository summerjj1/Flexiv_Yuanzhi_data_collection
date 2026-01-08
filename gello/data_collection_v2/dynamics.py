import xml.etree.ElementTree as ET
import numpy as np

# Rotation matrix from roll-pitch-yaw (extrinsic z-y-x)
def rpy_to_rot(roll, pitch, yaw):
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ])

# Rotation matrix from axis-angle
def axis_angle_to_rot(axis, theta):
    axis = np.array(axis) / np.linalg.norm(axis)
    ux, uy, uz = axis
    K = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

# Function to compute joint torques for quasi-static balance
def compute_gravity_torques(q, urdf_path='./src/urdf/force_gello_urdf_nofix.urdf', g_value=9.81):
    # Convert raw positions to radians
    # Assuming 0-4095 corresponds to 0-360 degrees, zero at 2048
    # radians = (q - 2048) * (2 * np.pi / 4096)
    q_rad = [(pos - 2048) * (2 * np.pi / 4096) for pos in q]

    # Parse URDF
    with open(urdf_path, 'r') as f:
        urdf_str = f.read()
    root = ET.fromstring(urdf_str)

    links = {}
    for link in root.findall('link'):
        name = link.get('name')
        inertial = link.find('inertial')
        if inertial is not None:
            orig = inertial.find('origin')
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
            if orig is not None:
                xyz = list(map(float, orig.get('xyz', '0 0 0').split()))
                rpy = list(map(float, orig.get('rpy', '0 0 0').split()))
            mass = float(inertial.find('mass').get('value'))
            links[name] = {'com_xyz': xyz, 'com_rpy': rpy, 'mass': mass}

    joints = {}
    for joint in root.findall('joint'):
        name = joint.get('name')
        j_type = joint.get('type')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        orig = joint.find('origin')
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if orig is not None:
            xyz = list(map(float, orig.get('xyz').split()))
            rpy = list(map(float, orig.get('rpy').split()))
        axis_elem = joint.find('axis')
        axis = [0.0, 0.0, 1.0]
        if axis_elem is not None:
            axis = list(map(float, axis_elem.get('xyz').split()))
        joints[name] = {'parent': parent, 'child': child, 'xyz': xyz, 'rpy': rpy, 'axis': axis, 'type': j_type}

    # Build kinematic chain
    link_order = ['base_link']
    joint_order = []
    current = 'base_link'
    while True:
        found = False
        for jname, j in joints.items():
            if j['parent'] == current:
                joint_order.append(jname)
                current = j['child']
                link_order.append(current)
                found = True
                break
        if not found:
            break

    # Forward kinematics
    assert len(q_rad) == len(joint_order), "Number of joint angles must match number of joints"

    g = np.array([0.0, 0.0, -g_value])  # Gravity vector

    T_world = [np.eye(4)]  # T for base_link
    joint_pos_list = []
    axis_world_list = []

    for i, jname in enumerate(joint_order):
        j = joints[jname]
        T_parent = T_world[-1]
        xyz = j['xyz']
        rpy = j['rpy']
        axis = np.array(j['axis'])
        theta = q_rad[i]

        R_fixed = rpy_to_rot(*rpy)
        T_fixed = np.eye(4)
        T_fixed[:3, :3] = R_fixed
        T_fixed[:3, 3] = xyz

        R_var = axis_angle_to_rot(axis, theta)
        T_var = np.eye(4)
        T_var[:3, :3] = R_var

        T_child = T_parent @ T_fixed @ T_var
        T_world.append(T_child)

        # Joint info
        T_joint = T_parent @ T_fixed
        joint_pos = T_joint[:3, 3]
        axis_world = T_joint[:3, :3] @ axis
        axis_world /= np.linalg.norm(axis_world)  # Normalize

        joint_pos_list.append(joint_pos)
        axis_world_list.append(axis_world)

    # Compute torques
    tau = np.zeros(len(q_rad))
    for i in range(len(q_rad)):
        sum_moment = np.zeros(3)
        for k in range(i + 1, len(link_order)):
            link_name = link_order[k]
            link = links[link_name]
            if link['mass'] == 0:
                continue
            # Assuming com_rpy is 0, as in the URDF
            com_xyz = np.array(link['com_xyz'])
            com_pos_world = (T_world[k] @ np.append(com_xyz, 1.0))[:3]
            F = link['mass'] * g
            r = com_pos_world - joint_pos_list[i]
            moment = np.cross(r, F)
            sum_moment += moment
        tau[i] = np.dot(axis_world_list[i], sum_moment)
    return tau

# Example usage
q = [2048.0] * 8  # All at zero position
torques = compute_gravity_torques(q)
print("Required joint torques to balance gravity:", torques)