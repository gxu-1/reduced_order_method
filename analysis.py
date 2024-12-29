# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:27:46 2024

@author: yirui
"""

import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from network_functions import PCAAutoencoder, TransformerAE,optimization_function,CustomTanh,Tanh
import functional_functions as ffc
#%% 加载数据
data_loaded = np.load('a337cabledata.npz')
# 将数据从 float64 转换为 float32 以节省内存
u = np.real(data_loaded['u']).astype(np.float32)
u0 = np.real(data_loaded['u0']).astype(np.float32)
du0 = 0.0001
t = np.real(data_loaded['t']).astype(np.float32)
sigma = np.real(data_loaded['fai']).astype(np.float32)
os.makedirs('FRC/', exist_ok=True)
os.makedirs('时程放大曲线/', exist_ok=True)

#%% 求解主导模态力
def Phi1(x, omega1):
    term1 = (2 * omega1 * np.cos(omega1 / 2) * np.cos(omega1 / 2)) / ((2 + np.cos(omega1)) * omega1 - 3 * np.sin(omega1))
    term2 = (1 - np.cos(omega1 * x) - np.tan(omega1 / 2) * np.sin(omega1 * x))
    return np.sqrt(term1) * term2

# 训练集train-yw
t1 = t[:]
yx1 = (u).astype(np.float32)

# 选取数据进行训练最开始3,2,2
steps = [2, 2]
valmax = 255
y3  = ffc.tiaoxuan(yx1,steps,valmax)
fai_3d = np.expand_dims(sigma, axis=-1)
# 再次使用 np.expand_dims 增加第二个维度
fai_3d = np.expand_dims(fai_3d, axis=-1)
fai  = ffc.tiaoxuan(fai_3d,steps,valmax)
fai = fai.reshape(-1, 1)
yy1, scaling_y1,scaling_y2 = ffc.scale_data(y3)

# 获取该数据的尺寸大小
ac = u.shape[0]
bc = u.shape[2]

# 计算原始数据的FRC曲线
sizes = math.ceil(bc * 0.5)
sizee = bc

# 对训练数据进行观察
fuzhi1 = ffc.HTtranslate(u[:,0,:],t,sizes,sizee,bc)

# 获取该数据的尺寸大小
ac1 = y3.shape[0]
bc1 = y3.shape[2]

# 计算原始数据的FRC曲线
sizes1 = math.ceil(bc1 * 0.5)
sizee1 = bc1

# 对训练数据进行观察
fuzhi2 = ffc.HTtranslate(y3[:,0,:],t1,sizes1,sizee1,bc1)
# 映射x
omegal = 9.47687626122814
f1 = 0.0002
fx_train = np.zeros((fai.shape[0],t1.shape[0]))
for i in range(fx_train.shape[0]):
    fx_train[i,:] = f1 * np.cos(t1 * (omegal + fai[i]))    
x1_trainz = fx_train.reshape(fx_train.shape[0], 1, fx_train.shape[1])
u0 = u0[:,np.newaxis,np.newaxis]
u1 = ffc.tiaoxuan(u0,steps,valmax)

xx1 = ((u1 + x1_trainz)).astype(np.float32)
dx1 = ((du0 + fx_train)).astype(np.float32)
# 记录代码结束的时间
# 绘图
plt.plot(sigma, fuzhi1, 'o', color='orange', label='original')
plt.plot(fai, fuzhi2, 'o', color='red', label='training')
plt.xlabel('Sigma')
plt.ylabel('Value')
plt.title('Solution of q1 over Sigma')
plt.legend(loc='upper right')
plt.savefig('FRC/FRC1.png', bbox_inches='tight', dpi=1000)
plt.show()

# 加载整个模型
def load_autoencoder(model_path):
    # 使用 torch.load() 加载模型
    model = torch.load(model_path)
    model.eval()  # 切换到评估模式
    return model

# 假设你的模型路径如下
model_path = 'model1.pth' 
model1 = load_autoencoder(model_path)

# 如果模型被 DataParallel 包装，先解除包装
if isinstance(model1, torch.nn.DataParallel):
    model1 = model1.module  
x1_tensor = torch.tensor(yy1, dtype=torch.float32)  
x1_tensor = x1_tensor.to('cpu') 

# 将模型和数据都放到 CPU 上
model1.to('cpu')
yl2,encoders = model1(x1_tensor)
yp1 = ffc.diweichonggou(yl2.detach().cpu().numpy(),scaling_y1,scaling_y2)
ty1 = ffc.inverse_scale(encoders.detach().cpu().numpy(),scaling_y1,scaling_y2)

# 对训练数据进行观察
fuzhi3 = ffc.HTtranslate(yp1[:,0,:],t1,sizes1,sizee1,bc1)

# 绘图
plt.plot(fai , fuzhi2, 'o', color='red', label='training')
plt.plot(fai , fuzhi3*1.1, 'o', color='green', label='predicted')
plt.legend(loc='upper right')
plt.xlabel('Sigma')
plt.ylabel('Value')
plt.title('FRC')
plt.savefig('FRC/FRC2.png', bbox_inches='tight', dpi=1000)
plt.show()

yy2 = yp1*1.1
yy,sacel1,secal2 = ffc.latent_data(yy2)
xx1 = xx1
xx,sacelx1,sacelx2 = ffc.latent_data(xx1)
# 假设你的模型路径如下
model_path = 'model2.pth'  
model2 = load_autoencoder(model_path)

if isinstance(model2, torch.nn.DataParallel):
    model2 = model2.module 
x1_tensor = torch.tensor(yy1, dtype=torch.float32)  
x1_tensor = x1_tensor.to('cpu') 
# 将模型和数据都放到 CPU 上
device = torch.device('cpu')
model2 = model2.to('cpu')  
x1_tensor = torch.tensor(xx, dtype=torch.float32) 
x1_tensor = x1_tensor.to('cpu')  
# 使用 TransformerAE 预测
with torch.no_grad():
     predictions = model2(torch.tensor(x1_tensor, dtype=torch.float), x1_tensor)

yp2 =  ffc.restore(predictions.detach().cpu().numpy(),sacel1,secal2)
# 超参数设置
learning_rate = 0.1  
num_iterations = 1000  
epsilon = 1e-7 

# Adam优化器的参数
beta1 = 0.9  
beta2 = 0.999  
epsilon_adam = 1e-8  
predicted_data = predictions.detach().cpu().numpy()
original_data = yy

# 调用优化函数
optimized_data, loss_history = optimization_function(
original_data, predicted_data, learning_rate, num_iterations, 
epsilon, beta1, beta2, epsilon_adam)

# 对训练数据进行观察
tyy = ffc.restore(optimized_data,sacel1,secal2)
fuzhi6 = ffc.HTtranslate(yy2,t1,sizes,sizee,bc)
fuzhi4 = ffc.HTtranslate(tyy,t1,sizes,sizee,bc)
# 绘图
plt.plot(fai, fuzhi3*1.1, 'o', color='orange', label='predicted')
plt.plot(fai, fuzhi4, 'o', color='green', label='optimized')
plt.plot(fai, fuzhi6, 'o', color='blue', label='real')
plt.legend(loc='upper right')
plt.xlabel('Sigma')
plt.ylabel('Value')
plt.title('FRC')
plt.savefig('FRC/FRC3.png', bbox_inches='tight', dpi=1000)
plt.show()
'''
# %%绘图保存
mask = (t1 >= 1996) & (t1 <= 1998)

predata  = tyy[55,0,:]
tradata  = y3[55,0,:]
# 使用 mask 来选择对应的 decoded_scaled 和 train_y_batch2
t1_subset = t1[mask]
pry1 = predata[mask]
ory = tradata[mask]

# 绘图
# 创建图像对象并设置图像大小
plt.figure(figsize=(10, 6))  
plt.plot(t1_subset, pry1, '*', color='red', label='predicted')
plt.plot(t1_subset, ory, '-', color='black', label='training')
#fig = plt.figure(figsize=(0.0018, 4))
plt.legend(loc='lower right')
plt.xlabel('time')
plt.ylabel('q(t)')
plt.title('Sigma(0.065)')
plt.savefig('时程放大曲线/timehistory.png', bbox_inches='tight', dpi=1000)
plt.show()
'''
omega1 = 9.47687626122814
x = 0.5
phi_result = Phi1(x, omega1)
#%% Sigma(0.065)
plt.figure(figsize=(10, 6))  
plt.plot(t1, tyy[55,0,:]*phi_result, '*', color='red', label='predicted')
plt.plot(t1, y3[55,0,:]*phi_result, '-', color='black', label='training')
# 设置横轴范围
plt.xlim(1996, 2000)
plt.ylim(-0.0022, 0.0030)  
plt.legend(loc='upper right')
plt.xlabel('time')
plt.ylabel('w(t)')
plt.title('Sigma(0.065)')
plt.savefig('时程放大曲线/Sigma(0.065).png', bbox_inches='tight', dpi=1000)
plt.show()
#%% Sigma(0.074)
plt.figure(figsize=(10, 6))  
plt.plot(t1, tyy[60,0,:]*phi_result, '*', color='red', label='predicted')
plt.plot(t1, y3[60,0,:]*phi_result, '-', color='black', label='training')
# 设置横轴范围
plt.xlim(1996, 2000)
plt.ylim(-0.0022, 0.0030)  
plt.legend(loc='upper right')
plt.xlabel('time')
plt.ylabel('w(t)')
plt.title('Sigma(0.074)')
plt.savefig('时程放大曲线/Sigma(0.074).png', bbox_inches='tight', dpi=1000)
plt.show()