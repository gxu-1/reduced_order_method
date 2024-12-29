# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:02:59 2024

@author: yirui
"""
import numpy as np
import torch
import torch.nn as nn


#%% 函数定义
class CustomTanh(nn.Module):
    def forward(self, x):
        return torch.tanh(x) * 1 
# 定义基于PCA的AutoEncoder
class PCAAutoencoder(nn.Module):
    def __init__(self, input_dim,hidden_size, latent_dim):
        super(PCAAutoencoder, self).__init__()
        
        # 编码器部分：将输入从 input_dim 降维到 latent_dim
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size//2),
            CustomTanh(), 
            nn.Linear(hidden_size//2, hidden_size//4),
            CustomTanh(),  
            nn.Linear(hidden_size//4, hidden_size//6),
            CustomTanh(), 
            nn.Linear(hidden_size//6, latent_dim),
            CustomTanh())                                                   

        # 解码器部分：将编码后的数据从 latent_dim 重建回 input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size//6),
            CustomTanh(),
            nn.Linear(hidden_size//6, hidden_size//4),
            CustomTanh(),
            nn.Linear(hidden_size//4, hidden_size//2),
            CustomTanh(),
            nn.Linear(hidden_size//2, hidden_size),
            CustomTanh(),
            nn.Linear(hidden_size, input_dim)) 
        
                    
    def forward(self, x1):
    
        x = x1.permute(0, 2, 1)  
        
        # 编码器
        encoded = self.encoder(x) 
        
        # 解码器
        decoded = self.decoder(encoded) 
        
        decoded = decoded.permute(0, 2, 1) 
        encoded1 = encoded.permute(0, 2, 1)
        
        return encoded1, decoded
    
    
class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        # 自定义 Tanh 激活函数
        return ((torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x)))*1.1

# 调用transformer训练
# Transformer-Encoder-Decoder模型
class TransformerAE(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=8, d_model=256, nhead=32, dim_feedforward=1024):
        super(TransformerAE, self).__init__()
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.activation = Tanh() 
        # Dropout layer
        self.dropout = nn.Dropout(p=0.4) 
        self.bn1 = nn.BatchNorm1d(d_model)

        self.fc_in = nn.Linear(input_dim, d_model)
        self.fc_out = nn.Linear(d_model, output_dim)
        
    def forward(self, src, tgt):
        src = self.fc_in(src)
        tgt = self.fc_in(tgt)
        
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        
        output = self.fc_out(output)
        return output
    
# 数据优化函数
# 均方误差（MSE）损失函数
def mse_loss(predicted, original):
    return np.mean((predicted - original) ** 2)

# 计算损失函数的梯度
def compute_gradient(predicted, original):
    return 2 * (predicted - original) / np.prod(predicted.shape)

# Adam优化器
def adam_optimizer(gradient, m, v, t, beta1, beta2, epsilon_adam, learning_rate):
    # 计算一阶矩和二阶矩的估计
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * gradient**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    
    # 更新参数
    update = -learning_rate * m_hat / (np.sqrt(v_hat) + epsilon_adam)
    
    return update, m, v

# 梯度下降优化过程（使用Adam优化器）
def optimization_function(original_data, predicted_data, learning_rate, num_iterations, epsilon, beta1, beta2, epsilon_adam):
    history = []  # 用来记录每次迭代的损失
    m = np.zeros_like(predicted_data)  # 一阶矩估计
    v = np.zeros_like(predicted_data)  # 二阶矩估计
    t = 0  # 迭代次数
    
    for iteration in range(num_iterations):
        # 计算当前的损失
        loss = mse_loss(predicted_data, original_data)
        history.append(loss)
        
        # 打印每次迭代的损失
        print(f"Iteration {iteration+1}/{num_iterations}, Loss: {loss}")
        
        # 计算梯度
        gradient = compute_gradient(predicted_data, original_data)
        
        # 使用Adam优化器更新参数
        t += 1
        update, m, v = adam_optimizer(gradient, m, v, t, beta1, beta2, epsilon_adam, learning_rate)
        
        # 更新预测数据
        predicted_data += update
        
        # 如果损失足够小，提前停止
        if loss < epsilon:
            print(f"Converged at iteration {iteration+1}")
            break
    
    return predicted_data, history
    
    