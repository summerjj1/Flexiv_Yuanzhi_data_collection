#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vive数据时间体系转换工具
将Vive原始时间戳转换到ToF时间体系下
作者: jiawei
"""

import os
import sys
import csv
import argparse
from datetime import datetime


class ViveOffsetPreprocessor:
    def __init__(self, session_dir: str):
        """
        初始化Vive时间体系转换工具
        
        Args:
            session_dir: 会话数据目录路径
        """
        self.session_dir = session_dir
        
        # 输入文件路径
        self.vive_original_file = os.path.join(session_dir, "Vive_Poses", "vive_data_tum.txt")
        self.slam_timestamps_file = os.path.join(session_dir, "SLAM_Poses", "slam_raw.txt")
        
        # 输出目录和文件
        self.output_dir = os.path.join(session_dir, "Vive_Poses_Offset")
        self.vive_offset_file = os.path.join(self.output_dir, "vive_data_tum.txt")
        self.offset_info_file = os.path.join(self.output_dir, "offset_info.txt")
        
        # 数据存储
        self.vive_first_ts = None
        self.tof_first_ts = None
        self.offset = None
        self.vive_data = []
        
        print("=" * 60)
        print("Vive数据时间体系转换工具")
        print("=" * 60)
        print(f"输入目录: {self.session_dir}")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)
    
    def check_input_files(self):
        """检查输入文件是否存在"""
        print("\n检查输入文件...")
        
        if not os.path.exists(self.vive_original_file):
            print(f"错误: Vive数据文件不存在: {self.vive_original_file}")
            sys.exit(1)
        
        if not os.path.exists(self.slam_timestamps_file):
            print(f"错误: SLAM时间戳文件不存在: {self.slam_timestamps_file}")
            sys.exit(1)
        
        print(f"  ✓ Vive原始数据: {self.vive_original_file}")
        print(f"  ✓ SLAM时间戳参考: {self.slam_timestamps_file}")
    
    def load_first_timestamps(self):
        """加载Vive和SLAM的第一帧时间戳"""
        print("\n加载第一帧时间戳...")
        
        # 读取Vive第一帧
        with open(self.vive_original_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                parts = first_line.split()
                if len(parts) >= 8:  # TUM格式: timestamp x y z qx qy qz qw
                    self.vive_first_ts = float(parts[0])
                else:
                    print(f"错误: Vive数据格式不正确: {first_line}")
                    sys.exit(1)
            else:
                print("错误: Vive数据文件为空")
                sys.exit(1)
        
        # 读取SLAM第一帧
        with open(self.slam_timestamps_file, 'r') as f:
            first_line = None
            for line in f:
                line = line.strip()
                if line:
                    first_line = line
                    break
            if first_line:
                parts = first_line.split()
                if len(parts) >= 8:
                    self.slam_first_ts = float(parts[0])
                else:
                    print(f"错误: SLAM数据格式不正确: {first_line}")
                    sys.exit(1)
            else:
                print("错误: SLAM数据文件为空")
                sys.exit(1)
        
        print(f"  Vive第一帧时间戳: {self.vive_first_ts:.6f}s")
        print(f"  SLAM第一帧时间戳: {self.slam_first_ts:.6f}s")
    
    def calculate_offset(self):
        """计算时间偏移量"""
        print("\n计算时间偏移量...")
        
        self.offset = self.slam_first_ts - self.vive_first_ts
        
        print(f"  计算的offset: {self.offset:.6f}s")
        print(f"  转换公式: vive_timestamp_aligned = vive_timestamp_original + ({self.offset:.6f})")
    
    def convert_vive_timestamps(self):
        """转换所有Vive时间戳"""
        print("\n转换Vive时间戳...")
        
        self.vive_data = []
        line_count = 0
        
        with open(self.vive_original_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 8:  # TUM格式: timestamp x y z qx qy qz qw
                    original_ts = float(parts[0])
                    
                    # 加上offset对齐到ToF时间体系
                    aligned_ts = original_ts + self.offset
                    
                    # 构建转换后的TUM格式行
                    aligned_line = f"{aligned_ts:.6f} {parts[1]} {parts[2]} {parts[3]} {parts[4]} {parts[5]} {parts[6]} {parts[7]}"
                    
                    self.vive_data.append({
                        'original_ts': original_ts,
                        'aligned_ts': aligned_ts,
                        'aligned_line': aligned_line
                    })
                    line_count += 1
                    
                    # 显示进度
                    if line_count % 500 == 0:
                        print(f"  已处理: {line_count} 条数据")
        
        print(f"  转换完成: 总共 {len(self.vive_data)} 条数据")
        
        # 显示时间范围
        if self.vive_data:
            first_aligned = self.vive_data[0]['aligned_ts']
            last_aligned = self.vive_data[-1]['aligned_ts']
            duration = last_aligned - first_aligned
            print(f"  转换后时间范围: {first_aligned:.6f}s ~ {last_aligned:.6f}s")
            print(f"  时长: {duration:.2f}秒")
    
    def save_results(self):
        """保存转换结果"""
        print("\n保存转换结果...")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存转换后的TUM文件
        with open(self.vive_offset_file, 'w') as f:
            for data in self.vive_data:
                f.write(f"{data['aligned_line']}\n")
        
        print(f"  ✓ 转换后数据已保存: {self.vive_offset_file}")
        
        # 保存offset信息
        with open(self.offset_info_file, 'w') as f:
            f.write("Vive数据时间体系转换信息\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("转换参数:\n")
            f.write(f"  Vive第一帧时间戳: {self.vive_first_ts:.6f}s\n")
            f.write(f"  参考源: SLAM\n")
            f.write(f"  SLAM第一帧时间戳: {self.slam_first_ts:.6f}s\n")
            f.write(f"  计算的offset: {self.offset:.6f}s\n")
            f.write(f"  转换公式: vive_ts_aligned = vive_ts_original + {self.offset:.6f}\n\n")
            
            f.write("转换统计:\n")
            f.write(f"  总共转换条目数: {len(self.vive_data)}\n")
            
            if self.vive_data:
                first_aligned = self.vive_data[0]['aligned_ts']
                last_aligned = self.vive_data[-1]['aligned_ts']
                duration = last_aligned - first_aligned
                f.write(f"  转换后时间范围: {first_aligned:.6f}s ~ {last_aligned:.6f}s\n")
                f.write(f"  时长: {duration:.2f}秒\n")
            
            f.write(f"\n转换时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"会话目录: {self.session_dir}\n")
        
        print(f"  ✓ 转换信息已保存: {self.offset_info_file}")
    
    def run(self):
        """运行完整的转换流程"""
        try:
            # 检查输入文件
            self.check_input_files()
            
            # 加载第一帧时间戳
            self.load_first_timestamps()
            
            # 计算offset
            self.calculate_offset()
            
            # 转换所有时间戳
            self.convert_vive_timestamps()
            
            # 保存结果
            self.save_results()
            
            print("\n" + "=" * 60)
            print("✓ Vive时间体系转换完成!")
            print(f"结果已保存到: {self.output_dir}")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Vive数据时间体系转换工具')
    parser.add_argument('session_dir', help='会话数据目录路径')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.session_dir):
        print(f"错误: 输入目录不存在: {args.session_dir}")
        sys.exit(1)
    
    # 创建转换工具并运行
    preprocessor = ViveOffsetPreprocessor(args.session_dir)
    preprocessor.run()


if __name__ == '__main__':
    main()
