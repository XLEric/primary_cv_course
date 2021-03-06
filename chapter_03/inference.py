#-*-coding:utf-8-*-
# date:2020-04-12
# Author: X.L.Eric
# function: inference

import os
import argparse
import torch
import torch.nn as nn
import numpy as np


import time
import datetime
import os
import math
from datetime import datetime
import cv2
import torch.nn.functional as F

from models.resnet_50 import resnet50
from models.my_model import MY_Net

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Project Classification Test')
    parser.add_argument('--test_model', type=str, default = './model_exp/2020-07-27_11-49-05/model_epoch-139.pth',
        help = 'test_model') # 模型路径
    parser.add_argument('--model', type=str, default = 'MY_Net',
        help = 'model : resnet_50,MY_Net') # 模型类型
    parser.add_argument('--num_classes', type=int , default = 200,
        help = 'num_classes') #  分类类别个数
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择
    parser.add_argument('--test_path', type=str, default = './example/',
        help = 'test_path') # 测试集路径
    parser.add_argument('--img_size', type=tuple , default = (256,256),
        help = 'img_size') # 输入模型图片尺寸
    parser.add_argument('--vis', type=bool , default = True,
        help = 'vis') # 是否可视化图片

    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    ops = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    print('----------------------------------')

    unparsed = vars(ops) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    #---------------------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

    test_path =  ops.test_path # 测试图片文件夹路径

    #---------------------------------------------------------------- 构建模型
    print('use model : %s'%(ops.model))

    if ops.model == 'resnet_50':
        model_ = resnet50(num_classes = ops.num_classes)
    else:
        model_ = MY_Net(num_classes = ops.num_classes)

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval() # 设置为前向推断模式

    # print(model_)# 打印模型结构

    # 加载测试模型
    if os.access(ops.test_model,os.F_OK):# checkpoint
        chkpt = torch.load(ops.test_model, map_location=device)
        model_.load_state_dict(chkpt)
        print('load test model : {}'.format(ops.test_model))

    #---------------------------------------------------------------- 预测图片
    font = cv2.FONT_HERSHEY_SIMPLEX
    with torch.no_grad():
        for file in os.listdir(ops.test_path):
            gt_label = file.split('_label_')[-1].strip('.jpg')
            print('------>>> {} - gt_label : {}'.format(file,gt_label))

            img = cv2.imread(ops.test_path + file)
            # 输入图片预处理
            img_ = cv2.resize(img, (ops.img_size[1],ops.img_size[0]))
            if ops.vis:
                cv2.namedWindow('image',0)
                cv2.imshow('image',img_)
                cv2.waitKey(1)
            img_ = img_.astype(np.float32)
            img_ = (img_-128.)/256.

            img_ = img_.transpose(2, 0, 1)
            img_ = torch.from_numpy(img_)
            img_ = img_.unsqueeze_(0)

            if use_cuda:
                img_ = img_.cuda()  # (bs, 3, h, w)

            pre_ = model_(img_.float())

            outputs = F.softmax(pre_,dim = 1)
            outputs = outputs[0]

            output = outputs.cpu().detach().numpy()
            output = np.array(output)

            max_index = np.argmax(output)

            score_ = output[max_index]

            print('gt {} -- pre {} : {}'.format(gt_label,max_index,score_))
            show_str = 'gt {} - pre {} :{:.2f}'.format(gt_label,max_index,score_)
            cv2.putText(img,show_str,(3,img.shape[0]-10),font,0.45,(15,125,255),3)
            cv2.putText(img,show_str,(3,img.shape[0]-10),font,0.45,(225,155,55),1)

            cv2.namedWindow('image',0)
            cv2.imshow('image',img)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    print('well done ')
