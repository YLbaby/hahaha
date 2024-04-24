import os
import random
from d2l import torch as d2l
import torch
from torch import nn

def load_files(directory):
    """加载指定目录下的所有文件名"""
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def split_data(files, train_ratio=0.7, dev_ratio=0.2):
    """根据给定的比例分割文件列表"""
    total_files = len(files)
    random.shuffle(files)
    train_end = int(total_files * train_ratio)
    dev_end = train_end + int(total_files * dev_ratio)
    train_files = files[:train_end]
    dev_files = files[train_end:dev_end]
    eval_files = files[dev_end:]
    return train_files, dev_files, eval_files

def write_to_file(file_list, file_name, spoof_label="spoof"):
    """将文件列表和对应标签写入到文件"""
    with open(file_name, 'a') as f:  # 使用 'a' 模式来追加数据
        for file in file_list:
            f.write(f"- {file} - - {spoof_label}\n")

def process_folder(parent_directory):
    """处理父目录下的所有子文件夹"""
    # 清空或创建文件
    open('train.txt', 'w').close()
    open('dev.txt', 'w').close()
    open('eval.txt', 'w').close()

    # 遍历所有子目录
    for folder in os.listdir(parent_directory):
        sub_dir = os.path.join(parent_directory, folder)
        if os.path.isdir(sub_dir):
            files = load_files(sub_dir)
            train_files, dev_files, eval_files = split_data(files)
            write_to_file(train_files, 'train.txt')
            write_to_file(dev_files, 'dev.txt')
            write_to_file(eval_files, 'eval.txt')

# 调用主函数
process_folder('Bert-VITS2')

# d2l.try_gpu