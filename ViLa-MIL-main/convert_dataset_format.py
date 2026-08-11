#!/usr/bin/env python3
"""
数据集格式转换脚本
将datasss.csv转换为ViLa-MIL项目要求的格式
"""

import pandas as pd
import os

def convert_dataset_format():
    """转换数据集格式"""
    
    # 读取原始CSV文件
    input_file = "datasss.csv"
    output_file = "dataset_csv/adenocarcinoma_dataset.csv"
    
    # 确保输出目录存在
    os.makedirs("dataset_csv", exist_ok=True)
    
    try:
        # 读取数据
        df = pd.read_csv(input_file)
        print(f"读取原始数据: {len(df)} 条记录")
        
        # 创建新的数据格式
        new_data = []
        
        for idx, row in df.iterrows():
            filename = row['filename']
            adenocarcinoma = row['Adenocarcinoma']
            
            # 根据Adenocarcinoma字段设置标签
            if adenocarcinoma == 1:
                label = "adenocarcinoma"
            else:
                label = "non_adenocarcinoma"
            
            # 创建case_id（使用patient_前缀）
            case_id = f"patient_{idx}"
            
            new_data.append({
                'case_id': case_id,
                'slide_id': filename,
                'label': label
            })
        
        # 创建新的DataFrame
        new_df = pd.DataFrame(new_data)
        
        # 保存到新文件
        new_df.to_csv(output_file, index=False)
        
        print(f"转换完成！")
        print(f"输出文件: {output_file}")
        print(f"转换后数据格式:")
        print(new_df.head())
        
        # 统计标签分布
        label_counts = new_df['label'].value_counts()
        print(f"\n标签分布:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
        
        # 如果您有非腺癌数据，需要添加到这个文件中
        if len(label_counts) == 1:
            print(f"\n注意: 当前只有腺癌数据，如需进行二分类，还需要添加非腺癌数据")
            print(f"请将非腺癌数据的Adenocarcinoma列设为0，然后重新运行此脚本")
            
    except Exception as e:
        print(f"转换失败: {e}")

if __name__ == "__main__":
    convert_dataset_format()
