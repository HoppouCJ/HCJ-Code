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
import os, sys


""" 读取数据重新划分"""







class DataSetIter(torch.utils.data.Dataset):
    def __init__(self, _base_img, _base_labels, _index2pos, _margin, _patch_size, _append_dim,_TE) -> None:
        self.base_img = _base_img #全量数据包括margin (145+2margin * 145+2margin * spe)
        self.base_labels = _base_labels #全量数据无margin (145 * 145)
        self.index2pos = _index2pos #训练数据 index -> (x, y) 对应margin后base_img的中心点坐标
        self.size = len(_index2pos)
        self.TE = _TE
        self.margin = _margin
        self.patch_size = _patch_size
        self.append_dim = _append_dim
    
    def __getitem__(self, index):
        start_x, start_y = self.index2pos[index]
        patch = self.base_img[start_x:start_x+2*self.margin+1 , start_y:start_y+2*self.margin+1,:]
        # sim = self.compute_block_similarity(patch, 8)
        # patch2 = sim[-1][0]
        # if self.append_dim:
        #     patch = np.expand_dims(patch, 0) # [channel=1, h, w, spe]
        #     patch = patch.transpose((0,3,1,2)) # [c, spe, h, w]
        # else:
        # if index%2 != 0:
        #     sim = self.compute_block_similarity(patch, 10)
        #     patch = sim[-1][0]
        patch = patch.transpose((2, 0, 1))
            # patch2 = patch2.transpose((2, 0, 1)) #[spe, h, w]
        if self.TE[start_x,start_y] == 0:
            label = self.base_labels[start_x, start_y] - 1
            
        else:
            label = torch.tensor(-1,dtype = torch.int64)
        # if index%2 !=0:
        #     patch = patch2

        # print(index, patch.shape, start_x, start_y, label)
        return torch.FloatTensor(patch), torch.LongTensor(label.reshape(-1))[0]  #将patch和标签组成元组返回，patch格式为FloatTensor，形状为(1，c,h,w),标签数据格式为LongTensor
    

    def __len__(self):
        return self.size
    
class DataSetIter_test(torch.utils.data.Dataset):
    def __init__(self, _base_img, _base_labels, _index2pos, _margin, _patch_size, _append_dim) -> None:
        self.base_img = _base_img #全量数据包括margin (145+2margin * 145+2margin * spe)
        self.base_labels = _base_labels #全量数据无margin (145 * 145)
        self.index2pos = _index2pos #训练数据 index -> (x, y) 对应margin后base_img的中心点坐标
        self.size = len(_index2pos)

        self.margin = _margin
        self.patch_size = _patch_size
        self.append_dim = _append_dim
    
    def __getitem__(self, index):
        start_x, start_y = self.index2pos[index]
        patch = self.base_img[start_x:start_x+2*self.margin+1 , start_y:start_y+2*self.margin+1,:]
        # sim = self.compute_block_similarity(patch, 8)
        # patch2 = sim[-1][0]
        # if self.append_dim:
        #     patch = np.expand_dims(patch, 0) # [channel=1, h, w, spe]
        #     patch = patch.transpose((0,3,1,2)) # [c, spe, h, w]
        # else:
        # if index%2 != 0:
        #     sim = self.compute_block_similarity(patch, 10)
        #     patch = sim[-1][0]
        patch = patch.transpose((2, 0, 1))
            # patch2 = patch2.transpose((2, 0, 1)) #[spe, h, w]
        
        label = self.base_labels[start_x, start_y] - 1
        # if index%2 !=0:
        #     patch = patch2

        # print(index, patch.shape, start_x, start_y, label)
        return torch.FloatTensor(patch), torch.LongTensor(label.reshape(-1))[0]  #将patch和标签组成元组返回，patch格式为FloatTensor，形状为(1，c,h,w),标签数据格式为LongTensor
    

    def __len__(self):
        return self.size








class HSIDataLoader(object):
    def __init__(self, param) -> None: #n为大patch为margin的n倍， m为取几个负样本
        # self.n = n
        # self.m = m
        self.data_param = param['data']
        self.data_path_prefix = "../data"
        self.data = None #原始读入X数据 shape=(h,w,c)
        self.labels = None #原始读入Y数据 shape=(h,w,1)
        self.TR = None #标记训练数据
        self.TE = None #标记测试数据

        # 参数设置
        self.if_numpy = self.data_param.get('if_numpy', False)
        self.data_sign = self.data_param.get('data_sign', 'Indian')
        self.patch_size = self.data_param.get('patch_size', 13) # n * n
        self.remove_zeros = self.data_param.get('remove_zeros', True)
        self.test_ratio = self.data_param.get('test_ratio', 0.9)
        self.batch_size = self.data_param.get('batch_size', 128)
        self.none_zero_num = self.data_param.get('none_zero_num', 0)
        self.spectracl_size = self.data_param.get("spectral_size", 0)
        self.append_dim = self.data_param.get("append_dim", False)
        self.use_norm = self.data_param.get("use_norm", True)
        self.perclass=self.data_param.get("perclass", 10)
        self.sample=self.data_param.get("sample", 1)
        self.weight=self.data_param.get("weight",'spe')


        self.diffusion_sign = self.data_param.get('diffusion_sign', False)
        self.diffusion_data_sign_path_prefix = self.data_param.get("diffusion_data_sign_path_prefix", '')
        self.diffusion_data_sign = self.data_param.get("diffusion_data_sign", "unet3d_27000.pkl")

    # def load_matdata(self, data_sign, data_path_prefix): #加载数据集
    #     if data_sign == "Indian":
    #         data = sio.loadmat('%s/Indian_pines_corrected.mat' % data_path_prefix)['indian_pines_corrected']
    #         labels = sio.loadmat('%s/Indian_pines_gt.mat' % data_path_prefix)['indian_pines_gt']
    #     elif data_sign == "Pavia":
    #         data = sio.loadmat('%s/PaviaU.mat' % data_path_prefix)['paviaU']
    #         labels = sio.loadmat('%s/PaviaU_gt.mat' % data_path_prefix)['paviaU_gt'] 
    #     elif data_sign == 'HongHu':
    #         data = sio.loadmat('%s/WHU_Hi_HongHu.mat' % data_path_prefix)['WHU_Hi_HongHu']
    #         labels = sio.loadmat('%s/WHU_Hi_HongHu_gt.mat' % data_path_prefix)['WHU_Hi_HongHu_gt']
    #     return data, labels

    # def gen(self, data_sign, train_num_per_class, data_path_prefix, max_percent=0.5):
    #     data, labels = self.load_matdata(data_sign, data_path_prefix)
    #     h, w, c = data.shape
    #     class_num = labels.max()
    #     class2data = {}
    #     for i in range(h):
    #         for j in range(w):
    #             if labels[i,j] > 0:
    #                 if labels[i, j] in class2data:
    #                     class2data[labels[i,j]].append([i, j])
    #                 else:
    #                     class2data[labels[i,j]] = [[i,j]]

    #     TR = np.zeros_like(labels)
    #     TE = np.zeros_like(labels)
    #     for cl in range(class_num):
    #         class_index = cl + 1
    #         ll = class2data[class_index]
    #         all_index = list(range(len(ll)))
    #         real_train_num = train_num_per_class
    #         if len(all_index) <= train_num_per_class:
    #             real_train_num = int(len(all_index) * max_percent) 
    #         select_train_index = set(random.sample(all_index, real_train_num))
    #         for index in select_train_index:
    #             item = ll[index]
    #             TR[item[0], item[1]] = class_index
    #     TE = labels - TR
    #     target = {}
    #     target['TE'] = TE
    #     target['TR'] = TR
    #     target['input'] = data
    #     return target

    # def run(self):
    #     # signs = ['Indian', 'Pavia',  'Salinas']
    #     signs = self.data_sign
    #     data_path_prefix = '../data/clipeverytime'
    #     train_num_per_class = self.perclass
    #     pp = '../data/clipeverytime/%s' % signs
    #     if not os.path.exists(pp):
    #         os.makedirs(pp)
        
    #     save_path = '%s/%s_%s_split.mat' %(pp, signs, train_num_per_class)
    #     target = self.gen(signs, train_num_per_class, data_path_prefix)
    #     sio.savemat(save_path, target)
    #     print('save %s done.' % save_path)


    def load_data_from_diffusion(self, data_ori, labels):
        path = "%s/%s" % (self.diffusion_data_sign_path_prefix, self.diffusion_data_sign)
        data = np.load(path)
        ori_h, ori_w, _= data_ori.shape
        h, w, _= data.shape
        assert ori_h == h, ori_w == w
        print("load diffusion data shape is ", data.shape)
        return data, labels 

    def load_raw_data(self):
        # self.run()
        data, labels = None, None
        assert self.data_sign in ['Indian', 'Pavia', 'Houston','Salinas', 'HongHu']
        # data_path = '%s/%s/%s_%d_split%d.mat' % (self.data_path_prefix, self.data_sign, self.data_sign,self.perclass,self.sample)
        data_path = '%s/%s/%s_%d_split.mat' % (self.data_path_prefix, self.data_sign, self.data_sign,self.perclass)
        all_data = sio.loadmat(data_path)
        data = all_data['input']
        TR = all_data['TR'] # train label
        TE = all_data['TE'] # test label
        labels = TR + TE
        return data, labels, TR, TE

    def load_data(self):
        ori_data, labels, TR, TE = self.load_raw_data()
        if self.diffusion_sign:
            diffusion_data, diffusion_labels = self.load_data_from_diffusion(ori_data, labels)
            return diffusion_data, diffusion_labels, TR, TE
        else:
            return ori_data, labels, TR, TE

    def _padding(self, X, margin=6):
        # pading with zeros
        w,h,c = X.shape
        new_x, new_h, new_c = w+margin*2, h+margin*2, c
        returnX = np.zeros((new_x, new_h, new_c))
        start_x, start_y = margin, margin
        returnX[start_x:start_x+w, start_y:start_y+h,:] = X
        return returnX
    
    def get_valid_num(self, y):
        tempy = y.reshape(-1)
        validy = tempy[tempy > 0]
        print('valid y shape is ', validy.shape)
        return validy.shape[0]

    def get_train_test_num(self, TR, TE):
        train_num, test_num = TR[TR>0].reshape(-1).size, TE[TE>0].reshape(-1).size
        print("train_num=%s, test_num=%s" % (train_num, test_num))
        return train_num, test_num

        
    def get_train_test_patches(self, X, y, TR, TE):
        h, w, c = X.shape
        i=0
        # 给 X 做 padding
        windowSize = self.patch_size
        margin = int((windowSize - 1) / 2)
        zeroPaddedX = self._padding(X, margin=margin)
        
        # 确定train和test的数据量
        train_num, test_num = self.get_train_test_num(TR, TE)
        trainX_index2pos = {}
        testX_index2pos = {}
        all_index2pos = {}

        patchIndex = 0
        trainIndex = 0
        testIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                start_x, start_y = r-margin, c-margin
                tempy = y[start_x, start_y]
                temp_tr = TR[start_x, start_y] 
                temp_te = TE[start_x, start_y]
                if temp_tr > 0 and temp_te > 0:
                    print("here", temp_tr, temp_te, r, c)
                    raise Exception("data error, find sample in trainset as well as testset.")

                if temp_tr > 0: #train data
                    for _ in range(4):
                        trainX_index2pos[trainIndex] = [start_x, start_y]
                        trainIndex += 1 #循环m次
                    
                    # trainX_index2pos[trainIndex] = [start_x, start_y]
                    # trainIndex += 1
                    # trainX_index2pos[trainIndex] = [start_x, start_y]
                    # trainIndex += 1
                

                elif temp_te > 0:
                    testX_index2pos[testIndex] = [start_x, start_y]
                    testIndex += 1
                    # testX_index2pos[testIndex] = [start_x, start_y]
                    # testIndex += 1
                all_index2pos[patchIndex] =[start_x, start_y]
                patchIndex = patchIndex + 1
                # all_index2pos[patchIndex] =[start_x, start_y]
                # patchIndex = patchIndex + 1
        return zeroPaddedX, y, trainX_index2pos, testX_index2pos, all_index2pos, margin, self.patch_size,train_num #返回扩展后的图像，目标标签，训练集、测试集、所有块像素块位置信息，填充大小和patch大小


    def applyPCA(self,   X, numComponents=30):
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
        return newX

    def data_preprocessing(self, data):
        '''
        1. normalization
        2. pca
        3. spectral filter
        data: [h, w, spectral]
        '''
        if self.use_norm:
            norm_data = np.zeros(data.shape)
            for i in range(data.shape[2]):
                input_max = np.max(data[:,:,i])
                input_min = np.min(data[:,:,i])
                norm_data[:,:,i] = (data[:,:,i]-input_min)/(input_max-input_min)
        else:
            norm_data = data 
        pca_num = self.data_param.get('pca', 0)
        if pca_num > 0:
            pca_data = self.applyPCA(norm_data, int(self.data_param['pca']))
            norm_data = pca_data
        if self.spectracl_size > 0: # 按照给定的spectral size截取数据
            norm_data = norm_data[:,:,:self.spectracl_size]
        return norm_data



    def generate_numpy_dataset(self):
        #1. 根据data_sign load data
        self.data, self.labels, self.TR, self.TE = self.load_data()
        print('[load data done.] load data shape data=%s, label=%s' % (str(self.data.shape), str(self.labels.shape)))

        #2. 数据预处理 主要是norm化
        norm_data = self.data_preprocessing(self.data) 
        
        print('[data preprocessing done.] data shape data=%s, label=%s' % (str(norm_data.shape), str(self.labels.shape))) 

        # 3. reshape & filter
        h, w, c = norm_data.shape
        norm_data = norm_data.reshape((h*w,c))
        norm_label = self.labels.reshape((h*w))
        TR_reshape = self.TR.reshape((h*w))
        TE_reshape = self.TE.reshape((h*w))
        TrainX = norm_data[TR_reshape>0]
        TrainY = norm_label[TR_reshape>0]
        TestX = norm_data[TE_reshape>0]
        TestY = norm_label[TE_reshape>0]
        train_test_data = norm_data[norm_label>0]
        train_test_label = norm_label[norm_label>0]
        
        print('------[data] split data to train, test------')
        print("X_train shape : %s" % str(TrainX.shape))
        print("Y_train shape : %s" % str(TrainY.shape))
        print("X_test shape : %s" % str(TestX.shape))
        print("Y_test shape : %s" % str(TestY.shape))

        return TrainX, TrainY, TestX, TestY, norm_data

    def reconstruct_pred(self, y_pred):
        '''
        根据原始label信息 对一维预测结果重建图像
        y_pred: [h*w]
        return: pred: [h, w]
        '''
        h, w = self.labels.shape
        return y_pred.reshape((h,w))
    

    def spe(self, vector1, vector2): #越接近0越相似
        vector1_1 = vector1.reshape(-1)
        vector2_1 = vector2.reshape(-1)
        cos_theta = np.dot(vector1_1,vector2_1)/(np.linalg.norm(vector1_1)*np.linalg.norm(vector2_1))
        sam = np.arccos(cos_theta)
        return sam

    def spa(self, vector1, vector2): #越接近1越相似
        ssim = compare_ssim(vector1, vector2,multichannel=True,data_range=vector2.max() - vector2.min()) #越接近0，空间相似度越低 
        # ssim = 1-ssim
        return ssim

    def norm(self, data):
        min_val = min(data)
        max_val = max(data)
        normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
        return normalized_data

    def power(self, spe,spa, weight):
        spe_norm = self.norm(spe)
        spa_norm = self.norm(spa)
        spe_part = [math.exp(weight * i) for i in spe_norm]
        spa_part = [math.exp((1 - weight) * j) for j in spa_norm]
        sim = [x*y for x,y in zip(spe_part, spa_part)] #指数函数法
        #接近1为负样本，接近0为正样本
        return sim


    # def compute_similarity(self,vector1, vector2, weight = 0.5): #对于负样本，要不同但相似，局部光谱相似度低，整体光谱相似度高，空间相似度高
    #     # print(vector1.shape)
    #     # print(vector2.shape)
    #     width, height = vector1.shape[0]//2
    #     vector1_small = vector1[width-2:width+3,height-2:width+3].reshape(-1)
    #     vector2_small = vector2[width-2:width+3,height-2:width+3].reshape(-1)
    #     vector1_1 = vector1.reshape(-1)
    #     vector2_1 = vector2.reshape(-1)
    #     # normalized_sid = SID(vector1_1, vector2_1)
    #     cos_theta = np.dot(vector1_1,vector2_1)/(np.linalg.norm(vector1_1)*np.linalg.norm(vector2_1))
    #     cos_theta_small = np.dot(vector1_small,vector2_small)/(np.linalg.norm(vector1_small)*np.linalg.norm(vector2_small))
    #     sam = np.arccos(cos_theta)
    #     sam_small = np.arccos(cos_theta_small)
    #     sam_norm = 1- (sam/np.pi) #越接近0，整体光谱相似度越低
    #     sam_small_norm = sam_small/np.pi #越接近0，局部光谱相似度越高
    #     sam_all = (sam_small_norm + sam_norm)/2
    #     ssim = compare_ssim(vector1, vector2,multichannel=True,data_range=vector2.max() - vector2.min()) #越接近0，空间相似度越低 
        
    #     sim = math.exp(weight * sam_all) * math.exp((1 - weight) * ssim) #指数函数法
        
    #     return sim
       
    def hn(self,train_index2pos,sim_train, train_num):
        for i in range(0,len(train_index2pos),4):
            for j in range(train_num):
                
                if train_index2pos[i] == sim_train[f'xy{j}']:
                    train_index2pos[i+1] = sim_train[f'sim{j}']
                    train_index2pos[i+2] = sim_train[f'sim2{j}']
                    train_index2pos[i+3] = sim_train[f'sim3{j}']
                        # for l in range(3):
                        #     data[i+l+1] = sim_all[f'sim{l+1}{j}'][0]
                        #     target[i+l+1] = sim_all[f'sim{l+1}{j}'][1]
                        # data[i+4] = sim_all[f'sim4{j}'][0]
                        # target[i+4] = sim_all[f'sim4{j}'][1]
                        # data[i+5] = sim_all[f'sim5{j}'][0]
                        # target[i+5] = sim_all[f'sim5{j}'][1]
            # combined = list(zip(data,target))
            # random.shuffle(combined)
            # data, target = zip(*combined)
        return train_index2pos


    def compute_hn(self, base_img,label,train_index2pos,margin,patch_size, train_num, TR):
        # base_img = self.applyPCA(base_img, 10)
        # print("base_img:",base_img.shape)
        sim_all ={}
        spe_list = []
        spa_list = []
        xy_list = []
        sim_positive = []
        sim_all["train_num"] = train_num
        l = 0
        if self.weight == 'spe':
            weight = 0
        if self.weight == 'spa':
            weight = 1
        if self.weight == 'ss':
            weight = 0.3
        
        for k in range(0,len(train_index2pos),4):#############################################################
            start_x, start_y = train_index2pos[k]
            patch0 = base_img[start_x:start_x+2*margin+1 , start_y:start_y+2*margin+1,:]
            target0 = label[start_x,start_y]-1
            target0 = torch.LongTensor(target0.reshape(-1))[0]
            # similarities= []
            for i in range(start_x - 4*margin , start_x+4*margin , 2):
                for j in range(start_y - 4*margin , start_y+4*margin, 2):
                    patch = base_img[i:i+patch_size , j:j+patch_size,:]
                    
                    # print(patch0.shape)
                    # # print(start_x,start_y)
                    if patch.shape != patch0.shape :
                        continue
                    target = label[i,j]-1
                    if target<0:
                        continue

                    # sim = self.compute_similarity(patch0,patch, weight)
                    sim_spe = self.spe(patch0, patch)
                    sim_spa = self.spa(patch0, patch)
                    sim_spa_fan = 1-sim_spa
                    patch = patch.transpose((2, 0, 1))
                    patch = torch.FloatTensor(patch)
                    spe_list.append(sim_spe)
                    spa_list.append(sim_spa)
                    if TR[i,j]-1 == target0:
                        sim_positive.append((sim_spe*sim_spa_fan,patch))

                    xy_list.append([i,j])
                    
                    # target = torch.LongTensor(target.reshape(-1))[0]
                    # similarities.append(([i, j],sim))
                    
            
            sim_list = self.power(spe_list, spa_list, 0.7)
            # print(sim_list)
            sim = [[x, y] for x, y in zip(xy_list, sim_list)]
            sim.sort(key=lambda x: x[1], reverse=True)
            sim_positive.sort(key=lambda x: x[0], reverse=True) #找到最不像的正样本
            # similarities.sort(key=lambda x: x[1], reverse=True)
            lenth=len(sim)
            # print(lenth)
            lenth=lenth//2
            patch0 = patch0.transpose((2, 0, 1))
            patch0 = torch.FloatTensor(patch0)
            sim_all[f'patch{l}'] = patch0
            sim_all[f'xy{l}'] = [start_x,start_y]
            sim_all[f'positive{l}'] = sim_positive[-1][1]
            sim_all[f'sim{l}'] = sim[1][0]
            sim_all[f'sim2{l}'] = sim[lenth//2][0]
            sim_all[f'sim3{l}'] = sim[lenth][0]
            l =l+1
        return sim_all
    
    def shuffle_position(self, train_index2pos, batch_size):
        
        keys = list(train_index2pos.keys())
        groups = [keys[i:i+4] for i in range(0, len(train_index2pos), 4)]
        random.shuffle(groups)
        shuffl_keys = [key for group in groups for key in group]
        groups_2 = [shuffl_keys[i:i+batch_size] for i in range(0, len(train_index2pos), batch_size)]
        shuffled_groups = [random.sample(group, len(group)) for group in groups_2]
        shuffled_keys = [key for group in shuffled_groups for key in group]
        shuffled_dict = {key: train_index2pos[key] for key in shuffled_keys}
        return shuffled_dict
    # def clip_dataset(self, dataset, batch_size):



    def generate_torch_dataset(self):
        # 0. 判断是否使用numpy数据集
        if self.if_numpy:
            return self.generate_numpy_dataset()

        #1. 根据data_sign load data
        self.data, self.labels, self.TR, self.TE = self.load_data()
        print('[load data done.] load data shape data=%s, label=%s' % (str(self.data.shape), str(self.labels.shape)))
        #2. 数据预处理 主要是norm化
        norm_data = self.data_preprocessing(self.data) 
        
        print('[data preprocessing done.] data shape data=%s, label=%s' % (str(norm_data.shape), str(self.labels.shape)))

        #3. 获取patch 并形成batch型数据
        base_img, labels, train_index2pos, test_index2pos, all_index2pos, margin, patch_size, train_num \
              = self.get_train_test_patches(norm_data, self.labels, self.TR, self.TE)
        
        # print(len(test_index2pos))
        # print(len(all_index2pos))
        # np.save('train_pos.npy',train_index2pos)
        # np.save('test_pos.npy',test_index2pos)
        # test_index2pos = np.load('test_pos.npy', allow_pickle=True).item()
        


        sim_train = self.compute_hn(base_img, labels,train_index2pos, margin, patch_size, train_num, self.TR)
        train_index2pos = self.hn(train_index2pos, sim_train, train_num)
        # # print(train_index2pos)
        # # for key in test_index2pos:
        # #     if key in train_index2pos:
        # #         keys_to_delete.append(key)
        # # print(len(keys_to_delete))
        # # for key in keys_to_delete:
        # #     del test_index2pos[key]
        
        
        np.save('test1.npy',sim_train)
        train_index2pos = self.shuffle_position(train_index2pos, self.batch_size)
        # print(train_index2pos)


       
        

        # np.save('train_pos(replaced).npy',train_index2pos)
        # train_index2pos = np.load('train_pos.npy', allow_pickle=True).item()
        
        # print(sim_train['sim20'][1])

        # sim_unlabel = self.compute_hn(base_img, train_index2pos, margin)
        
        # print("保存成功")
        # sim_all = np.load('sim1.npy', allow_pickle=True).item()

        print('------[data] split data to train, test------')
        print("train len: %s" % len(train_index2pos ))
        print("test len : %s" % len(test_index2pos ))
        print("all len: %s" % len(all_index2pos ))

        multi=self.data_param.get('unlabelled_multiple',1)

        trainset = DataSetIter(base_img, labels, train_index2pos, margin, patch_size, self.append_dim, self.TE) 
        unlabelset=DataSetIter_test(base_img,labels,test_index2pos,margin, patch_size, self.append_dim)
        testset = DataSetIter_test(base_img, labels, test_index2pos , margin, patch_size, self.append_dim) 
        allset = DataSetIter_test(base_img, labels, all_index2pos, margin, patch_size, self.append_dim) 
        # new_trainset = self.hn(trainset)
        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                drop_last=False
                                                )
        unlabel_loader=torch.utils.data.DataLoader(dataset=unlabelset,
                                                batch_size=int(self.batch_size*multi),
                                                shuffle=True,
                                                num_workers=0,
                                                drop_last=False)
        test_loader = torch.utils.data.DataLoader(dataset=testset,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=False
                                                )
        all_loader = torch.utils.data.DataLoader(dataset=allset,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=False
                                                )
        # dataloader = self.test(train_loader,sim_all)
        # train_loader1 = self.hn(train_loader, sim_train)
        # print("train替换完成")

        # unlabel_loader = self.hn(unlabel_loader,sim_all)
        # print("unlabel替换完成")
        # train_loader = self.hn(train_loader)
        return train_loader, unlabel_loader,test_loader, all_loader
        # return trainset,unlabelset,testset,allset

       


if __name__ == "__main__":
    dataloader = HSIDataLoader({"data":{}})
    train_loader, unlabel_loader, test_loader, all_loader = dataloader.generate_torch_dataset()
    def read(data):
        data0 = data.numpy()
        data0 = np.transpose(data0,(1,2,0))
        view = spy.imshow(data0)
    def shift(data):
        b, s, h, w = data.shape
        delta = torch.FloatTensor(1, 1, 1).uniform_(-0.1, 0.1)
        # expanded_delta = delta_list.expand_as(data)
        shifted_data = [image - torch.randn_like(image) * 0.1 for image in data]
        return shifted_data
    
    data, target = next(iter(train_loader))
    read(data[0])
    plt.pause(60)
    shifted_data = shift(data)
    data0 = shifted_data[0]
    read(data0)
    plt.pause(60)

    
    # data, target = next(iter(train_loader))
    # print(target)
    # print("打乱前",train_index2pos)
    # groups = [train_index2pos[i:i+128] for i in range(0,len(train_index2pos),128)]
    # for group in groups:
    #     random.shuffle(group)
    # print("打乱后",train_index2pos)




    
   
