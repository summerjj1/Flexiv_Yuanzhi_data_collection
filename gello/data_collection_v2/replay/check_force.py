import h5py
import numpy as np
import matplotlib.pyplot as plt

def visualize_6d_force_simple(h5_file_path, dataset_name='f_ext_base_frame', 
                              start_idx=0, end_idx=None):
    """
    简洁的6D力数据可视化
    第一行三张子图：fx, fy, fz (力)
    第二行三张子图：tx, ty, tz (力矩)
    """
    
    # 读取H5文件
    with h5py.File(h5_file_path, 'r') as f:
        data = f[dataset_name][:]
    
    # 截取指定范围
    if end_idx is None:
        end_idx = len(data)
    data = data[start_idx:end_idx]
    n_samples = len(data)
    time = np.arange(n_samples)
    
    # 提取6个分量
    fx = data[:, 0]
    fy = data[:, 1]
    fz = data[:, 2]
    tx = data[:, 3]
    ty = data[:, 4]
    tz = data[:, 5]
    
    # 创建图形
    fig = plt.figure(figsize=(15, 8))
    
    # 第一行：力分量
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(time, fx, 'r-', linewidth=1.5)
    ax1.set_xlabel('采样点')
    ax1.set_ylabel('Fx (N)')
    ax1.set_title('X方向力')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(time, fy, 'g-', linewidth=1.5)
    ax2.set_xlabel('采样点')
    ax2.set_ylabel('Fy (N)')
    ax2.set_title('Y方向力')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(time, fz, 'b-', linewidth=1.5)
    ax3.set_xlabel('采样点')
    ax3.set_ylabel('Fz (N)')
    ax3.set_title('Z方向力')
    ax3.grid(True, alpha=0.3)
    
    # 第二行：力矩分量
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(time, tx, 'c-', linewidth=1.5)
    ax4.set_xlabel('采样点')
    ax4.set_ylabel('Tx (N·m)')
    ax4.set_title('X方向力矩')
    ax4.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(time, ty, 'm-', linewidth=1.5)
    ax5.set_xlabel('采样点')
    ax5.set_ylabel('Ty (N·m)')
    ax5.set_title('Y方向力矩')
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(time, tz, 'y-', linewidth=1.5)
    ax6.set_xlabel('采样点')
    ax6.set_ylabel('Tz (N·m)')
    ax6.set_title('Z方向力矩')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'机械臂6D力数据 (f_ext_base_frame) - 共{n_samples}个采样点', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 打印基本统计信息
    print("力数据统计信息：")
    print(f"Fx: 均值={np.mean(fx):.3f} N, 标准差={np.std(fx):.3f}, 范围=[{np.min(fx):.3f}, {np.max(fx):.3f}]")
    print(f"Fy: 均值={np.mean(fy):.3f} N, 标准差={np.std(fy):.3f}, 范围=[{np.min(fy):.3f}, {np.max(fy):.3f}]")
    print(f"Fz: 均值={np.mean(fz):.3f} N, 标准差={np.std(fz):.3f}, 范围=[{np.min(fz):.3f}, {np.max(fz):.3f}]")
    print(f"Tx: 均值={np.mean(tx):.3f} N·m, 标准差={np.std(tx):.3f}, 范围=[{np.min(tx):.3f}, {np.max(tx):.3f}]")
    print(f"Ty: 均值={np.mean(ty):.3f} N·m, 标准差={np.std(ty):.3f}, 范围=[{np.min(ty):.3f}, {np.max(ty):.3f}]")
    print(f"Tz: 均值={np.mean(tz):.3f} N·m, 标准差={np.std(tz):.3f}, 范围=[{np.min(tz):.3f}, {np.max(tz):.3f}]")
    
    return data

# 使用示例
if __name__ == "__main__":
    # 替换为您的H5文件路径
    # h5_file = "/media/ubuntu/ZhqSSD/YZdata/insert_hole_threecam/2026-01-08/trajectory_20260108_133243.h5" #trajectory_20260108_133015.h5
    h5_file = "/media/ubuntu/ZhqSSD/YZdata/insert_hole_threecam/2026-01-08/trajectory_20260108_133015.h5" #
    
    # 可视化数据
    try:
        data = visualize_6d_force_simple(
            h5_file_path=h5_file,
            dataset_name='f_ext_base_frame',
            start_idx=0,
            end_idx=None  # 设为None可显示全部数据
        )
    except Exception as e:
        print(f"错误: {e}")
        print("请检查：")
        print("1. 文件路径是否正确")
        print("2. 数据集中是否有 'f_ext_base_frame' 这个key")