import h5py

def print_h5_keys_and_shapes(file_path):
    with h5py.File(file_path, 'r') as f:
        # 遍历文件中的所有键名
        def print_group(name, obj):
            if isinstance(obj, h5py.Dataset):  # 只打印数据集
                print(f"Key: {name}, Shape: {obj.shape}")
                # import pdb;pdb.set_trace()  # 设置断点
        # 遍历文件中的所有对象和键
        f.visititems(print_group)

# 使用方法
file_path = '/home/vla/code/data/2025-12-10/trajectory_20251210_232010.h5'  # 替换为你的文件路径
print_h5_keys_and_shapes(file_path)
