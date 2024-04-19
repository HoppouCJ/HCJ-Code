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


""" Training dataset"""







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
        self.data_path_prefix = "./../../data"
        self.data = None #原始读入X数据 shape=(h,w,c)
        self.labels = None #原始读入Y数据 shape=(h,w,1)
        self.TR = None #标记训练数据
        self.TE = None #标记测试数据

        # 参数设置
        self.if_numpy = self.data_param.get('if_numpy', False)
        self.data_sign = self.data_param.get('data_sign', 'Pavia')
        self.patch_size = self.data_param.get('patch_size', 13) # n * n
        self.remove_zeros = self.data_param.get('remove_zeros', True)
        self.test_ratio = self.data_param.get('test_ratio', 0.9)
        self.batch_size = self.data_param.get('batch_size', 128)
        self.none_zero_num = self.data_param.get('none_zero_num', 0)
        self.spectracl_size = self.data_param.get("spectral_size", 0)
        self.append_dim = self.data_param.get("append_dim", False)
        self.use_norm = self.data_param.get("use_norm", True)
        self.perclass=self.data_param.get("perclass", 50)
        self.sample=self.data_param.get("sample", 1)
        self.weight=self.data_param.get("weight",'spe')


        self.diffusion_sign = self.data_param.get('diffusion_sign', False)
        self.diffusion_data_sign_path_prefix = self.data_param.get("diffusion_data_sign_path_prefix", '')
        self.diffusion_data_sign = self.data_param.get("diffusion_data_sign", "unet3d_27000.pkl")

    def load_data_from_diffusion(self, data_ori, labels):
        path = "%s/%s" % (self.diffusion_data_sign_path_prefix, self.diffusion_data_sign)
        data = np.load(path)
        ori_h, ori_w, _= data_ori.shape
        h, w, _= data.shape
        assert ori_h == h, ori_w == w
        print("load diffusion data shape is ", data.shape)
        return data, labels 

    def load_raw_data(self):
        data, labels = None, None
        assert self.data_sign in ['Indian', 'Pavia', 'Houston','Salinas']
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
                    trainX_index2pos[trainIndex] = [start_x, start_y]
                    trainIndex += 1
                    # trainX_index2pos[trainIndex] = [start_x, start_y]
                    # trainIndex += 1
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
    

    


    def compute_similarity(self,vector1, vector2, weight = 0.5):
        # print(vector1.shape)
        # print(vector2.shape)
        vector1_1 = vector1.reshape(-1)
        vector2_1 = vector2.reshape(-1)
        # normalized_sid = SID(vector1_1, vector2_1)
        cos_theta = np.dot(vector1_1,vector2_1)/(np.linalg.norm(vector1_1)*np.linalg.norm(vector2_1))
        sam = np.arccos(cos_theta)
        sam_norm = 1-(sam/np.pi)
        # sigma = 1.0  # 高斯核函数的带宽参数
        # distance = np.linalg.norm(vector1_1 - vector2_1)  # 计算样本之间的欧几里得距离
        # kernel_value = np.exp(-distance**2 / (2 * sigma**2))
        # distance = np.linalg.norm(vector1_1 - vector2_1)
        # max_distance = np.sqrt(np.sum(np.square(np.max(vector1_1)), axis=-1))
        # normalized_distance = distance / max_distance
        ssim = compare_ssim(vector1, vector2,multichannel=True,data_range=vector2.max() - vector2.min()) 
        # sim = weight*ssim+(1-weight)*sam_norm #加权平均
        sim = math.exp(weight * sam_norm) * math.exp((1 - weight) * ssim) #指数函数法
        # sim = sam_norm*ssim #乘法法
        return sim
       
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


    def compute_hn(self, base_img,label,train_index2pos,margin,patch_size, train_num):
        # base_img = self.applyPCA(base_img, 10)
        # print("base_img:",base_img.shape)
        sim_all ={}
        sim_all["train_num"] = train_num
        l = 0
        if self.weight == 'spe':
            weight = 0
        if self.weight == 'spa':
            weight = 1
        if self.weight == 'ss':
            weight = 0.5
        print(weight)
        for k in range(0,len(train_index2pos),4):#############################################################
            start_x, start_y = train_index2pos[k]
            patch = base_img[start_x:start_x+2*margin+1 , start_y:start_y+2*margin+1,:]
            target0 = label[start_x,start_y]-1
            target0 = torch.LongTensor(target0.reshape(-1))[0]
            similarities= []
            for i in range(start_x - 4*margin , start_x+4*margin , 4):
                for j in range(start_y - 4*margin , start_y+4*margin, 4):
                    patch0 = base_img[i:i+patch_size , j:j+patch_size,:]
                    
                    # print(patch0.shape)
                    # # print(start_x,start_y)
                    if patch0.shape != patch.shape :
                        continue
                    # target = label[i,i]-1
                    
                    # if target<0:
                    #     continue
                    # sim = self.compute_similarity(patch0,patch, weight)
                    patch0 = patch0.transpose((2, 0, 1))
                    patch0 = torch.FloatTensor(patch0)
                    # target = torch.LongTensor(target.reshape(-1))[0]
            #         similarities.append(([i, j],sim))
                    
            
            # similarities.sort(key=lambda x: x[1], reverse=True)
            # lenth=len(similarities)
            # lenth=lenth//2
            patch = patch.transpose((2, 0, 1))
            patch = torch.FloatTensor(patch)
            sim_all[f'patch{l}'] = patch, target0
            # sim_all[f'xy{l}'] = [start_x,start_y]
            # sim_all[f'sim{l}'] = similarities[-1][0]
            # sim_all[f'sim2{l}'] = similarities[-lenth//2][0]
            # sim_all[f'sim3{l}'] = similarities[-lenth][0]
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
        
        center_x,center_y = train_index2pos[50]
        patch = base_img[]
        # np.save('train_pos.npy',train_index2pos)
        # np.save('test_pos.npy',test_index2pos)
        # test_index2pos = np.load('test_pos.npy', allow_pickle=True).item()
        


        # sim_train = self.compute_hn(base_img, labels,train_index2pos, margin, patch_size, train_num)
        # train_index2pos = self.hn(train_index2pos, sim_train, train_num)
        
        # for key in test_index2pos:
        #     if key in train_index2pos:
        #         keys_to_delete.append(key)
        # print(len(keys_to_delete))
        # for key in keys_to_delete:
        #     del test_index2pos[key]
        
        
        # np.save('test1.npy',sim_train)
        # train_index2pos = self.shuffle_position(train_index2pos, self.batch_size)


       
        

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

        trainset = DataSetIter_test(base_img, labels, train_index2pos, margin, patch_size, self.append_dim) 
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
    train_loader, unlabel_loader, test_loader, all_loader= dataloader.generate_torch_dataset()
    def read(data):
        data0 = data.numpy()
        data0 = np.transpose(data0,(1,2,0))
        view = spy.imshow(data0)
        plt.pause(0.1)
    for data, target in train_loader:
        for i in range(128):
            print(data[i])
    # for data, target in train_loader:
    #     for i in data
    #     print(data[0].shape)
   
    # for i in range(64):
    #     data0 = trainset[i][0].numpy()
    #     if np.all(trainset[i][0] == 0) == "True":
    #         print('cao')
    # def read(data):
    #     data0 = data.numpy()
    #     data0 = np.transpose(data0,(1,2,0))
    #     view = spy.imshow(data0)
    
    # print("打乱前",train_index2pos)
    # groups = [train_index2pos[i:i+128] for i in range(0,len(train_index2pos),128)]
    # for group in groups:
    #     random.shuffle(group)
    # print("打乱后",train_index2pos)