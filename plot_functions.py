# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:08:50 2024

@author: yirui
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl



matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Times New Roman'
def plot(train_y_batch, output_seq, t, epoch):
    fig = plt.figure(figsize=(8, 8))
    # Ensure t is detached and converted to numpy array if it's a tensor
    if isinstance(t, torch.Tensor):
        t = t.cpu().detach().numpy()
     # 确保 t 是一个 numpy 数组
    if isinstance(t, torch.Tensor):
        t = t.cpu().detach().numpy()   

    output_seq = output_seq.cpu().detach().numpy()

    train_y_batch = train_y_batch.cpu().detach().numpy()
    
    # Plot with t as the x-axis
    plt.plot(t, train_y_batch[0, :, :].reshape(-1), label='Actual')

    plt.plot(t, output_seq[0, :, :].reshape(-1), color='orange', label='Training')
    
    plt.xlabel('t')
    plt.ylabel('q(t)')
    plt.legend()
    plt.title(f'Epoch {epoch}')
    return fig    


mpl.rcParams['agg.path.chunksize'] = 10000  # 根据需要调整这个数值
def plot1(train_y_batch, output_seq, t, epoch):
    fig = plt.figure(figsize=(8, 8))
    # Ensure t is detached and converted to numpy array if it's a tensor
    if isinstance(t, torch.Tensor):
        t = t.detach().numpy()
    
    if isinstance(train_y_batch, torch.Tensor):
        train_y_batch = train_y_batch.cpu().detach().numpy()
     # 确保 t 是一个 numpy 数组
    if isinstance(output_seq, torch.Tensor):
        output_seq = output_seq.cpu().detach().numpy()   
    # 因为 t 的大小是 (13334,)，因此只绘制每个样本中的一个序列
    plt.plot(t, train_y_batch[0, 0, :], label='Actual')  
    plt.plot(t, output_seq[0, 0, :], color='orange', label='Training') 
    
    plt.xlabel('t')
    plt.ylabel('q(t)')
    plt.legend()
    plt.title(f'Epoch {epoch}')
    return fig  
def plot2(train_y_batch, output_seq, t, epoch):
    fig = plt.figure(figsize=(8, 8))
    # Ensure t is detached and converted to numpy array if it's a tensor
    if isinstance(t, torch.Tensor):
        t = t.detach().numpy()
    
    if isinstance(train_y_batch, torch.Tensor):
        train_y_batch = train_y_batch.cpu().detach().numpy()
     # 确保 t 是一个 numpy 数组
    if isinstance(output_seq, torch.Tensor):
        output_seq = output_seq.cpu().detach().numpy()  
        
    #print(f"train_y_batch:{train_y_batch.shape}")  
    #print(f"output_seq:{output_seq.shape}")  
    # 因为 t 的大小是 (13334,)，因此只绘制每个样本中的一个序列
    plt.plot(t, train_y_batch[0, 0, :], label='Actual')  # 绘制第一个 batch 的第一个序列
    plt.plot(t, output_seq[0, 0, :], color='orange', label='Training')  # 同样绘制第一个 batch 的第一个序列
    
    plt.xlabel('t')
    plt.ylabel('q(t)')
    plt.legend()
    plt.title(f'Epoch {epoch}')
    return fig    
'''
def plot_original_data_snapshot(features, num_samples=4, save_dir=None, epoch=None):
    """
    绘制原始数据快照，保留数据轮廓
    :param features: 输入数据，形状为 (batch_size, features, sequence_length)
    :param num_samples: 选择前几个样本进行绘制
    :param save_dir: 保存图像的目录（如果为 None，则直接显示图像）
    :param epoch: 当前训练轮次，用于文件命名
    """
    # 检查数据类型并转换为 PyTorch 张量
    if isinstance(features, np.ndarray):
        features = torch.tensor(features)

    # 检查数据形状
    if features.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got shape {features.shape}")

    batch_size, num_features, seq_length = features.shape

    # 确保样本数不超过 batch_size
    num_samples = min(num_samples, batch_size)

    # 动态计算行数和列数
    rows = (num_samples + 1) // 2  # 计算所需行数，2 列布局
    cols = min(2, num_samples)  # 列数最多为 2

    # 创建绘图
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)

    # 绘制特征图
    for i in range(num_samples):
        # 提取单个样本的特征图 (num_features, seq_length)
        feature_map = features[i].cpu().numpy()

        # 将特征图转换为可视化矩阵
        reshaped_feature = feature_map.T  # 转置方便可视化 (seq_length, num_features)

        # 绘制特征图（热图显示）
        row, col = divmod(i, cols)  # 计算子图位置
        axes[row][col].imshow(reshaped_feature, aspect='auto', cmap='viridis')
        axes[row][col].set_title(f"Original Data {i+1}")
        axes[row][col].axis("on")  # 显示坐标轴

    # 调整布局
    plt.tight_layout()

    # 保存或显示图片
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = f"original_data_epoch_{epoch}.png" if epoch is not None else "original_data.png"
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Original data snapshots saved to {save_path}")
    else:
        plt.show()
'''        
def plot_combined_features(features, num_samples=4, save_dir=None, epoch=None, downsample_factor=50):
    """
    叠加原始数据和像素化特征图
    :param features: 输入数据，形状为 (batch_size, features, sequence_length)
    :param num_samples: 选择前几个样本进行绘制
    :param save_dir: 保存图像的目录（如果为 None，则直接显示图像）
    :param epoch: 当前训练轮次，用于文件命名
    :param downsample_factor: 数据下采样的因子
    """
    # 检查数据类型并转换为 PyTorch 张量
    if isinstance(features, np.ndarray):
        features = torch.tensor(features)

    # 检查数据形状
    if features.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got shape {features.shape}")

    batch_size, num_features, seq_length = features.shape

    # 确保样本数不超过 batch_size
    num_samples = min(num_samples, batch_size)

    # 动态计算行数和列数
    rows = (num_samples + 1) // 2  # 计算所需行数，2 列布局
    cols = min(2, num_samples)  # 列数最多为 2

    # 创建绘图
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)

    # 绘制特征图
    for i in range(num_samples):
        # 提取单个样本的特征图 (num_features, seq_length)
        feature_map = features[i].cpu().numpy()

        # 原始数据
        original_feature = feature_map.T  # 转置 (seq_length, num_features)
        downsampled_feature = original_feature[::downsample_factor, ::downsample_factor]

        # 绘制特征图
        row, col = divmod(i, cols)  # 计算子图位置
        axes[row][col].imshow(original_feature, aspect='auto', cmap='viridis', alpha=0.5)  # 原始数据背景
        axes[row][col].imshow(downsampled_feature, aspect='auto', cmap='hsv', interpolation='nearest', alpha=0.7)  # 稀疏特征
        axes[row][col].set_title(f"Combined Feature Map {i+1}")
        axes[row][col].axis("off")  # 关闭坐标轴

    # 调整布局
    plt.tight_layout()

    # 保存或显示图片
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = f"combined_features_epoch_{epoch}.png" if epoch is not None else "combined_features.png"
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Combined feature maps saved to {save_path}")
    else:
        plt.show()        