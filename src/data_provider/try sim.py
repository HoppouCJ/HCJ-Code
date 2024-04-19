import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import random
import torch.optim as optim
from operator import truediv
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
import time
import matplotlib.pyplot as plt
import spectral as spy
import cv2
import math
data_path = "./../../data/Indian/Indian_10_split.mat"
all_data = sio.loadmat(data_path)
data = all_data['input']
TR = all_data['TR'] # train label
TE = all_data['TE'] # test label
labels = TR + TE
# print(data.shape)
center_x, center_y = data.shape[0]//2-20, data.shape[1]//2+20
print(center_x, center_y)
patch = data[center_x-6:center_x+7,center_y-6:center_y+7,:]
print(labels[center_x,center_y])
def norm(list):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    
    return normalized_data 

def compute_similarity(vector1, vector2, weight = 0.5): #对于负样本，要不同但相似，局部光谱相似度低，整体光谱相似度高，空间相似度高
        # print(vector1.shape)
        # print(vector2.shape)
        width, height = vector1.shape[0]//2, vector1.shape[1]//2
        # vector1_small = vector1[width-2:width+3,height-2:height+3,:].reshape(-1)
        # vector2_small = vector2[width-2:width+3,height-2:height+3,:].reshape(-1)
        vector1_1 = vector1.reshape(-1)
        vector2_1 = vector2.reshape(-1)
        # normalized_sid = SID(vector1_1, vector2_1)
        cos_theta = np.dot(vector1_1,vector2_1)/(np.linalg.norm(vector1_1)*np.linalg.norm(vector2_1))
        # cos_theta_small = np.dot(vector1_small,vector2_small)/(np.linalg.norm(vector1_small)*np.linalg.norm(vector2_small))
        sam = np.arccos(cos_theta)
        # sam_small = np.arccos(cos_theta_small)
        # sam_norm = norm(sam) #越接近0，整体光谱相似度越高
        # sam_small_norm = sam_small/np.pi #越接近0，局部光谱相似度越高
        # sam_all = (sam_small_norm + sam_norm)/2
        ssim = compare_ssim(vector1, vector2,multichannel=True,data_range=vector2.max() - vector2.min()) #越接近0，空间相似度越低 
        # ssim_norm = norm(ssim)
        sim = math.exp(weight * sam) * math.exp((1 - weight) * ssim) #指数函数法
        #接近1为负样本，接近0为正样本
        return sim

def spe(vector1, vector2): #越接近0越相似
    vector1_1 = vector1.reshape(-1)
    vector2_1 = vector2.reshape(-1)
    cos_theta = np.dot(vector1_1,vector2_1)/(np.linalg.norm(vector1_1)*np.linalg.norm(vector2_1))
    sam = np.arccos(cos_theta)
    return sam

def spa(vector1, vector2): #越接近1越相似
    ssim = compare_ssim(vector1, vector2,multichannel=True,data_range=vector2.max() - vector2.min()) #越接近0，空间相似度越低 
    # ssim = 1-ssim
    return ssim

def norm(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

def power(spe,spa, weight):
    spe_norm = norm(spe)
    spa_norm = norm(spa)
    spe_part = [math.exp(weight * i) for i in spe_norm]
    spa_part = [math.exp((1 - weight) * j) for j in spa_norm]
    sim = [x*y for x,y in zip(spe_part, spa_part)] #指数函数法
        #接近1为负样本，接近0为正样本
    return sim

def padding(X, margin=6):
        # pading with zeros
        w,h,c = X.shape
        new_x, new_h, new_c = w+margin*2, h+margin*2, c
        returnX = np.zeros((new_x, new_h, new_c))
        start_x, start_y = margin, margin
        returnX[start_x:start_x+w, start_y:start_y+h,:] = X
        return returnX

def compute(patch, x, y, baseimg, baselabel, TR):
    similarities= []
    spe_list = []
    spa_list = []
    target_list = []
    sim_positive = []
    target0 = 9
    for i in range(x - 4*6 , x+4*6+1 , 2):
         for j in range(y - 4*6 , y+4*6+1, 2):
                patch2 =  baseimg[i:i+13 , j:j+13,:]
                if patch2.shape != patch.shape :
                    continue
                target = baselabel[i,j]-1
                if target<0:
                    continue
                
                    
                # sim = compute_similarity(patch,patch2, 0.7)
                # similarities.append((target, sim))
                sim_spe = spe(patch, patch2)
                sim_spa = spa(patch, patch2)
                sim_spa_fan = 1-sim_spa
                spe_list.append(sim_spe)
                spa_list.append(sim_spa)
                if TR[i,j] == 10:
                   sim_positive.append((sim_spe*sim_spa_fan,target))

                target_list.append(target)
    
    sim_list = power(spe_list, spa_list, 0.7)
    # print(sim_list)
    sim = [[x, y] for x, y in zip(target_list, sim_list)]
    sim.sort(key=lambda x: x[1], reverse=True)
    sim_positive.sort(key=lambda x: x[0], reverse=True) #找到最不像的正样本
    return sim_positive

pad_img = padding(data,6)
sim = compute(patch, center_x, center_y, pad_img,labels, TR)

for i in range(len(sim)):
      print(sim[i][1])

# for i in range(428):
#     print(labels[sim[i][0]])
