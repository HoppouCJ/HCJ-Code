import os, sys, time, json
import numpy as np
import time
import utils
from utils import recorder

from data_provider.data_provider import HSIDataLoader 
from trainer import get_trainer, BaseTrainer, CrossTransformerTrainer
import evaluation
import spectral as spy
import matplotlib.pyplot as plt
# from torchsummary import summary
# from torchstat import stat
# from ptflops import get_model_complexity_info

DEFAULT_RES_SAVE_PATH_PREFIX = "./res/"

def train_by_param(param):
    #0. recorder reset防止污染数据
    recorder.reset()
    # 1. 数据生成
    dataloader = HSIDataLoader(param)
    train_loader,unlabel_loader, test_loader, all_loader = dataloader.generate_torch_dataset()
    # sim_train = np.load('test1.npy', allow_pickle=True).item()
    # print(sim_train.keys())
    # data, target = next(iter(train_loader))
    # print(data.shape)
    
    # 2. 训练和测试
    trainer = get_trainer(param)
    trainer.train(train_loader, unlabel_loader,test_loader)
    # summary(trainer.net,[(100,13,13)]) 
    # print(stat(trainer.net, (100,13,13)))
    # flops, params = get_model_complexity_info(trainer.net, (100,13,13), as_strings=True, print_per_layer_stat=True)
    # print(flops)
    # print(params)
    # exit()
    # eval_res = trainer.final_eval(all_loader)
    # pred_all, y_all = trainer.test(all_loader)
    # pred_matrix = dataloader.reconstruct_pred(pred_all)

    #3. record all information
    recorder.record_param(param)
    # recorder.record_eval(eval_res)
    # recorder.record_pred(pred_matrix)

    return recorder

def train_convention_by_param(param):
    #0. recorder reset防止污染数据
    recorder.reset()
    # 1. 数据生成
    
    dataloader = HSIDataLoader(param)
    trainX, trainY, testX, testY, allX = dataloader.generate_torch_dataset() 

    # 2. 训练和测试
    trainer = get_trainer(param)
    trainer.train(trainX, trainY)
    eval_res = trainer.final_eval(testX, testY)
    pred_all = trainer.test(allX)
    pred_matrix = dataloader.reconstruct_pred(pred_all)

    #3. record all information
    recorder.record_param(param)
    recorder.record_eval(eval_res)
    recorder.record_pred(pred_matrix)

    return recorder 




include_path = [
    # 'houston_contra_mask.json',
    'indian_contra_mask.json'
    # ,
    # 'pavia_contra_mask.json'
    # 'salinas_contra_mask.json'
    # 'HongHu_contra_mask.json'
    # 'pavia_SSRN.json',
    # 'indian_SSRN.json',
    # 'salinas_SSRN.json'
    # 'houston_SSRN.json'

]

def check_convention(name):
    for a in ['knn', 'random_forest', 'svm']:
        if a in name:
            return True
    return False

def run_all():
    save_path_prefix = './res/GGCL/'
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    for name in include_path:
        convention = check_convention(name)
        path_param = './params/%s' % name
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
        print('start to train %s...' % name)
    # for n in range(4,6):
    #     for m in range(4,7,2):
        for _ in range(5): #总体跑2次
            if convention:
                train_convention_by_param(param)
            else:
                train_by_param(param)
            print('model eval done of %s...' % param['data']['data_sign'])
        # 保存路径，首先添加compared模型的名称
            path=save_path_prefix+param['net']['trainer']+'/'
            if not os.path.exists(path):
                os.mkdir(path)
        # 添加dataset
            path=path+param['data']['data_sign']+'/'
            if not os.path.exists(path):
                os.mkdir(path)
        # 添加perclass
            path=path+str(param['data']['perclass'])+'/'
            if not os.path.exists(path):
                os.mkdir(path)
        #添加sample
            path=path+str(param['train']['use_unlabel'])+'/'
            if not os.path.exists(path):
                os.mkdir(path)
            path=path+str(param['data']['weight'])+'0.3'+'/'
            if not os.path.exists(path):
                os.mkdir(path)
            recorder.to_file(path)


if __name__ == "__main__":
    run_all()
    