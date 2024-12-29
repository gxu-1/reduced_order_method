# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 08:28:42 2024

@author: yirui
"""

import os
import math
import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from scipy import integrate
import train_functions as tfs
import functional_functions as ffc
from torch.utils.data import DataLoader, TensorDataset
from network_functions import PCAAutoencoder, TransformerAE,optimization_function
#%% 设置设备
os.makedirs('FRC/', exist_ok=True)
os.makedirs('时程放大曲线/', exist_ok=True)
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
os.makedirs('FRC/', exist_ok=True)
# 低维空间转换系数
lamda = 1.1
# %%%
# 记录代码开始的时间
start_time = time.time()
#%% 加载数据
data_loaded = np.load('a337cabledata.npz', mmap_mode='r')
# 将数据从 float64 转换为 float32 以节省内存
u = np.real(data_loaded['u']).astype(np.float32)
du = np.real(data_loaded['du']).astype(np.float32)
u0 = np.real(data_loaded['u0']).astype(np.float32)
du0 = 0.001
t = np.real(data_loaded['t']).astype(np.float32)
sigma = np.real(data_loaded['fai']).astype(np.float32)

# 获取该数据的尺寸大小
ac = u.shape[0]
bc = u.shape[2]

# 计算原始数据的FRC曲线
sizes = math.ceil(bc * 0.5)
sizee = bc
ux =u[:,0,]
fuzhi1 = ffc.HTtranslate(ux,t,sizes,sizee,bc)
# 绘图
plt.plot(sigma, fuzhi1, 'o', color='orange', label='original')
plt.xlabel('Sigma')
plt.ylabel('Value')
plt.title('Solution of q1 over Sigma')
plt.legend(loc='upper right')
plt.savefig('FRC/FRC1.png', bbox_inches='tight', dpi=1000)
plt.show()
#%% 求解主导模态力
def Phi1(x, omega1):
    term1 = (2 * omega1 * np.cos(omega1 / 2) * np.cos(omega1 / 2)) / ((2 + np.cos(omega1)) * omega1 - 3 * np.sin(omega1))
    term2 = (1 - np.cos(omega1 * x) - np.tan(omega1 / 2) * np.sin(omega1 * x))
    return np.sqrt(term1) * term2
# 定义积分函数
def integrate_Phi1(omega1, x_min=0, x_max=1):
    # 对Phi1(x)进行积分，返回积分结果
    result, error = integrate.quad(lambda x: Phi1(x, omega1), x_min, x_max)
    return result

# omega1范围从1到10
omega_values = [4.213685, 9.47687, 15.71855, 21.99493,28.276101,34.558483, 40.841287,47.12426, 53.40733,59.69044]

# 记录每个omega1值对应的积分结果
integral_results = []

for omega1 in omega_values:
    integral_value = integrate_Phi1(omega1)
    integral_results.append(integral_value)

# 将结果转换为numpy数组
integral_results = np.array(integral_results)
    
omegal = 9.47687626122814
f1 = 0.000601977
j1 = np.array((integral_results))
# 初始化矩阵
fx_train = np.zeros((sigma.shape[0],j1.shape[0],t.shape[0]))
for i in range(sigma.shape[0]):
    for j in range(len(j1)):
        fx_train[i,j,:] = j1[j]* f1 * np.cos(t * (omegal + sigma[i]))    
u0 = u0[:,np.newaxis,np.newaxis]
x11 = ((u0 + fx_train)).astype(np.float32)
dx1 = ((du0 + fx_train)).astype(np.float32)

#%% 筛选数据
steps = [2,2]  # 定义步长
valmax = 255
xt  = ffc.tiaoxuan(x11,steps,valmax)
xdt = ffc.tiaoxuan(dx1,steps,valmax)
yt  = ffc.tiaoxuan(u,steps,valmax)
ydt = ffc.tiaoxuan(du,steps,valmax)
fai_3d = np.expand_dims(sigma, axis=-1)
fai_3d = np.expand_dims(fai_3d, axis=-1)
fai  = ffc.tiaoxuan(fai_3d,steps,valmax)
phi = fai.reshape(-1, 1)
# 对训练数据进行观察
fuzhi2 = ffc.HTtranslate(yt[:,0,:],t,sizes,sizee,bc)
# 绘图
plt.plot(sigma , fuzhi1, 'o', color='red', label='original')
plt.plot(phi , fuzhi2, 'o', color='green', label='training')
plt.xlabel('Sigma')
plt.ylabel('Value')
plt.title('FRC')
plt.legend(loc='upper right')
plt.savefig('FRC/FRC2.png', bbox_inches='tight', dpi=1000)
plt.show()
# 组合u和du
x2 = np.concatenate((xt,xdt),axis=0)

# 对数据进行归一化
x1, scaling_x1,scaling_x2 = tfs.scale_data(x2)
x1 = np.expand_dims(x1[:, 0, :], axis=1)
# 训练集train-yw
t1 = t[:]
y11 = (yt).astype(np.float32)
dy1 = (ydt).astype(np.float32)
y2 = np.concatenate((y11,dy1),axis=0)
y1, scaling_y1,scaling_y2 = tfs.scale_data(y2)
# %% 定义autoencoder网络参数
input_dim = 10  
latent_dim = 1 
hidden_size = 512
num_epoch1 = 10001 
batch_size1 = 8
model1 = PCAAutoencoder(input_dim,hidden_size,latent_dim)
model1.to(device)
model1 = torch.nn.DataParallel(model1)
optimizer1 = optim.Adam(model1.parameters(), lr=0.0001, weight_decay=0.0001)
scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.1, patience=10, verbose=True)
loss_all1 = []
valloss_all1 = []        

# %% 训练autoencoder
os.makedirs('降维/', exist_ok=True)
os.makedirs('重构/', exist_ok=True)
os.makedirs('低维模型预测/', exist_ok=True)
os.makedirs('低维模型重构/', exist_ok=True)
decodes_times=[]
for epoch in range(num_epoch1):
    epoch_start_time = time.time()  # 每次迭代开始时记录时间
    idx = list(range(0, y1.shape[0]))
    random.shuffle(idx)
    k = math.ceil(y1.shape[0] * 0.8)
    idx_train1 = idx[:k]
    idx_val1 = idx[k:]
    
    train_y1 = torch.tensor(y1[np.array(idx_train1)], dtype=torch.float)
    test_y1 = torch.tensor(y1[np.array(idx_val1)], dtype=torch.float)
    train_x1 = torch.tensor(x1[np.array(idx_train1)], dtype=torch.float)
    idx_val = np.array(idx_val1, dtype=int)  # 确保 idx_val 是整数数组
    test_x1= torch.tensor(x1[np.array(idx_val1)], dtype=torch.float)
    
    train_data1 = TensorDataset(train_x1,train_y1)  # 使用输入数据作为目标
    test_data1 = TensorDataset(test_x1,test_y1)  # 使用输入数据作为目标
    train_loader1 = DataLoader(train_data1, batch_size=batch_size1, shuffle=True)
    test_loader1 = DataLoader(test_data1, batch_size=batch_size1, shuffle=True)
    
    # 训练autoencoder
    model1, trainloss, valloss, lr_autoencoder = tfs.train_autoencoder(train_loader1, test_loader1, model1, optimizer1, scheduler1,epoch,num_epoch1,t1)
    epoch_time = time.time() - epoch_start_time
    decodes_times.append(epoch_time)   
    print(f"Time: {epoch_time:.4f}s")

# 获取model1中的数据
if isinstance(model1, torch.nn.DataParallel):
    model1 = model1.module  # 解除 DataParallel 包装
model1.to('cpu')
yp = torch.tensor(y1, dtype=torch.float)
yp = yp.to('cpu') 
yl2,encoders = model1(yp)
yl3 = ffc.diweichonggou(yl2.detach().cpu().numpy(),scaling_y1,scaling_y2)
ty1 = ffc.inverse_scale(encoders.detach().cpu().numpy(),scaling_y1,scaling_y2)
shapa = yl3.shape[0]
acx = shapa // 2  
uxx1 = yl3[:acx ,:,:]
# 对训练数据进行观察
fuzhi3 = ffc.HTtranslate(uxx1,t,sizes,sizee,bc)
# 绘图
plt.plot(sigma, fuzhi1, 'o', color='red', label='original')
plt.plot(phi, fuzhi3*lamda, 'o', color='green', label='predicted')
plt.legend(loc='upper right')
plt.xlabel('Sigma')
plt.ylabel('Value')
plt.title('FRC')
plt.savefig('FRC/FRC3.png', bbox_inches='tight', dpi=1000)
plt.show()
#%% 设置transformer参数  
# 设置模型参数
input_size2 = 20000
d_model = 256
nhead = 16
dim_feedforward = 512
num_layers = 14
batch_size2 = 8
num_epoch2 = 40001
model2 =  TransformerAE(input_dim=input_size2, output_dim=input_size2, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=num_layers)
transformer_optimizer = torch.optim.Adam(model2.parameters(), lr=0.0005, weight_decay=0.0001)
# 定义学习率调度器（可选）
transformer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(transformer_optimizer, mode='min', factor=0.1, patience=10, verbose=True)
model2.to(device)
model2  = torch.nn.DataParallel(model2)
yy2 = yl3*lamda
yy,sacel1,secal2 = ffc.latent_data(yy2)
xx1 = x2[:,0,:]
xx1 = np.expand_dims(xx1, axis=1)
xx,sacelx1,sacelx2 = ffc.latent_data(xx1)

# 需要讨论tyyh与yl2是否差距很大
for epoch in range(num_epoch2):
    epoch_start_time = time.time()  # 每次迭代开始时记录时间
    idx = list(range(0, yy2.shape[0]))
    random.shuffle(idx)
    k = math.ceil(yy2.shape[0] * 0.8)
    idx_train3 = idx[:k]
    idx_val3 = idx[k:]    
    train_xd3 = torch.tensor(xx[np.array(idx_train3)], dtype=torch.float)
    test_xd3 = torch.tensor(xx[np.array(idx_val3)], dtype=torch.float)
    train_yg3 = torch.tensor(yy[np.array(idx_train3)], dtype=torch.float)
    idx_val = np.array(idx_val3, dtype=int)  # 确保 idx_val 是整数数组
    test_yg3 = torch.tensor(yy[np.array(idx_val3)], dtype=torch.float)
    train_data3 = TensorDataset(train_xd3,train_yg3)  # 使用输入数据作为目标
    test_data3 = TensorDataset(test_xd3,test_yg3)  # 使用输入数据作为目标
    train_loader3 = DataLoader(train_data3, batch_size=batch_size1, shuffle=True)
    test_loader3 = DataLoader(test_data3, batch_size=batch_size1, shuffle=True)
    model2, decoder_loss_all, val_decoder_loss_all, lr_reduction_history = tfs.train_transformer(train_loader3, test_loader3, model2, transformer_optimizer, transformer_scheduler,epoch, num_epoch2, t1)
    epoch_time = time.time() - epoch_start_time                                                
    decodes_times.append(epoch_time)   
    print(f"Time: {epoch_time:.4f}s")
# %%获取数据及数据优化    
if isinstance(model2, torch.nn.DataParallel):
    model2 = model2.module  # 解除 DataParallel 包装

# 将模型和数据都放到 CPU 上
model2 = model2.to('cpu')  # 确保模型在 CPU 上
x1_tensor = torch.tensor(xx, dtype=torch.float32)  # 转换为 PyTorch 张量
x1_tensor = x1_tensor.to('cpu')  # 确保输入数据在 CPU 上

# 使用 TransformerAE 预测
with torch.no_grad():
     predictions = model2(torch.tensor(x1_tensor, dtype=torch.float), x1_tensor)

yp2 =  tfs.restore(predictions.detach().cpu().numpy(),sacel1,secal2)
# 超参数设置
learning_rate = 0.1  
num_iterations = 1000  
epsilon = 1e-7  

# Adam优化器的参数
beta1 = 0.9  # 一阶矩估计的衰减率
beta2 = 0.999  # 二阶矩估计的衰减率
epsilon_adam = 1e-8  
predicted_data = predictions.detach().cpu().numpy()
original_data = yy

# 调用优化函数
optimized_data, loss_history = optimization_function(
original_data, predicted_data, learning_rate, num_iterations, 
epsilon, beta1, beta2, epsilon_adam)

# 对训练数据进行观察
tyy = ffc.restore(optimized_data,sacel1,secal2)
fuzhi5 = ffc.HTtranslate(yy2,t1,sizes,sizee,bc)
fuzhi4 = ffc.HTtranslate(tyy,t1,sizes,sizee,bc)
fuzhi4 = fuzhi4[:acx]
fuzhi5 = fuzhi5[:acx]
# 绘图
plt.plot(phi, fuzhi3*lamda, 'o', color='orange', label='predicted')
plt.plot(phi, fuzhi4, 'o', color='green', label='optimized')
plt.plot(phi, fuzhi5, 'o', color='blue', label='training')
plt.legend(loc='upper right')
plt.xlabel('Sigma')
plt.ylabel('Value')
plt.title('FRC')
plt.savefig('FRC/FRC3png', bbox_inches='tight', dpi=1000)
plt.show()
if tyy.ndim == 3:
    pass  # 什么都不做
else:
    tyy = np.expand_dims(tyy, axis=1)
  
model1.to(device)
model1 = torch.nn.DataParallel(model1)
tyy1 = tyy/lamda
# 恢复autoencoder，encoder数据
tyyd = ffc.diweifanchonggou(tyy1,scaling_y1,scaling_y2)
yhg = y1[:,:,:]

batch_size3 = 8
num_epoch3 = 10001
# 需要讨论tyyh与yl2是否差距很大
for epoch in range(num_epoch3):
    epoch_start_time = time.time()  # 每次迭代开始时记录时间
    idx = list(range(0, ty1.shape[0]))
    random.shuffle(idx)
    k = math.ceil(ty1.shape[0] * 0.8)
    idx_train3 = idx[:k]
    idx_val3 = idx[k:]    
    train_yd3 = torch.tensor(tyyd[np.array(idx_train3)], dtype=torch.float)
    test_yd3 = torch.tensor(tyyd[np.array(idx_val3)], dtype=torch.float)
    train_yg3 = torch.tensor(yhg[np.array(idx_train3)], dtype=torch.float)
    idx_val = np.array(idx_val3, dtype=int)  # 确保 idx_val 是整数数组
    test_yg3 = torch.tensor(yhg[np.array(idx_val3)], dtype=torch.float)
    train_data3 = TensorDataset(train_yd3,train_yg3)  # 使用输入数据作为目标
    test_data3 = TensorDataset(test_yd3,test_yg3)  # 使用输入数据作为目标
    train_loader3 = DataLoader(train_data3, batch_size=batch_size1, shuffle=True)
    test_loader3 = DataLoader(test_data3, batch_size=batch_size1, shuffle=True)
    model1, decoder_loss_all, val_decoder_loss_all, lr_reduction_history = tfs.train_decoder(train_loader3, test_loader3, model1,epoch, num_epoch2,t1)
    epoch_time = time.time() - epoch_start_time
    decodes_times.append(epoch_time)   
    print(f"Time: {epoch_time:.4f}s")

# %% 获取预测数据
# 如果模型被 DataParallel 包装，先解除包装
if isinstance(model1, torch.nn.DataParallel):
    model1 = model1.module  # 解除 DataParallel 包装
model1.to('cpu')
decoders = model1.decoder(torch.tensor(tyyd, dtype=torch.float))
ty1 = ffc.inverse_scale(decoders.detach().cpu().numpy(),scaling_y1,scaling_y2)

#%% 计算w(x,t)
results = []
x = 0.5
for omega1 in omega_values:
    result = Phi1(x, omega1)
    results.append(result)

# 将结果转换为numpy数组，便于后续处理
values_at_x_tensor = np.array(results)
values_at_x_tensor = values_at_x_tensor.unsqueeze(0).unsqueeze(2)  
# 进行元素乘法
decoded_scaled = ty1 * values_at_x_tensor     
train_y_batch2 = yt * values_at_x_tensor     
# 对第二维（维度1，即每个样本的 10 个数据）求和，得到形状 (8, 1, 10000)
decoded_scaled = decoded_scaled.sum(dim=1, keepdim=True)  # 在第二维上求和，保留维度
train_y_batch2 = train_y_batch2.sum(dim=1, keepdim=True)  # 在第二维上求和，保留维度      
'''  
# %%绘图保存
mask = (t1 >= 1996) & (t1 <= 1998)

predata  = decoded_scaled[60,:,:]
tradata  = train_y_batch2[60,:,:]
# 使用 mask 来选择对应的 decoded_scaled 和 train_y_batch2
t1_subset = t1[mask]
pry1 = predata[mask]
ory = tradata[mask]

# 绘图
plt.plot(t1_subset, pry1, '*', color='orange', label='predicted')
plt.plot(t1_subset, ory, '-', color='blue', label='training')
plt.legend(loc='best')
plt.xlabel('Sigma')
plt.ylabel('Value')
plt.title('FRC')
plt.savefig('时程放大曲线/sigma1.png', bbox_inches='tight', dpi=1000)
plt.show()
'''