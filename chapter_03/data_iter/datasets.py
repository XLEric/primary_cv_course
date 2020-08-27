#-*-coding:utf-8-*-
# date:2019-05-20
# Author: X.L.Eric
# function: data iter
import glob
import math
import os
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class LoadImagesAndLabels(Dataset):
    def __init__(self, path, img_size=(256,256),vis = False):
        print('img_size (height,width) : ',img_size[0],img_size[1])
        labels_ = []
        files_ = []
        for idx,doc in enumerate(sorted(os.listdir(path), key=lambda x:int(x.split('.')[0]), reverse=False)):
            print(' %s label is %s \n'%(doc,idx))
            for file in os.listdir(path+doc):
                if '.jpg' in file :# 同时过滤掉 val 数据集
                    labels_.append(idx)
                    files_.append(path+doc + '/' + file)
            print()
        print('\n')
        self.labels = labels_ # 样本标签获取
        self.files = files_ # 样本图片路径获取
        self.img_size = img_size# 图像尺寸参数获取
        self.vis = vis # 可视化参数获取
    def __len__(self):
        return len(self.files)#返回数据集的长度
    def __getitem__(self, index):
        img_path = self.files[index]# 获得索引样本对应的图片路径
        label_ = self.labels[index]# 获得索引样本对应的标签号
        img = cv2.imread(img_path)  # BGR 格式
        img_ = cv2.resize(img, (self.img_size[1],self.img_size[0]))# 图像统一尺寸
        if random.random()>0.5:# 数据扩增
            img_ = cv2.flip(img_, 1)# 左右翻转
        if self.vis:
            cv2.putText(img_,str(label_),(3,img_.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(15,125,255),2)
            cv2.putText(img_,str(label_),(3,img_.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,25),1)
            cv2.namedWindow('image',0)
            cv2.imshow('image',img_)
            cv2.waitKey(1)
        img_ = img_.astype(np.float32)
        img_ = (img_-128.)/256.# 数据预处理 ： 归一化
        img_ = img_.transpose(2, 0, 1)#转为 pytorch的数据格式，（height，width，channel）-》（ channel ，height，width）
        return img_,label_ # 返回图像预处理数据、标签

if __name__ == "__main__":
    train_path = './datasets/train_datasets/'
    img_size = (224,224)
    dataset = LoadImagesAndLabels(path = train_path,img_size = img_size,vis = True)
    print('len train datasets : %s'%(dataset.__len__()))
    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=1,
                            shuffle=True,
                            pin_memory=False,
                            drop_last = True)

    for epoch in range(0, 1):
        for i, (imgs_, labels_) in enumerate(dataloader):
            print('imgs size {} , labels size {}'.format(imgs_.size(), labels_.size()))
