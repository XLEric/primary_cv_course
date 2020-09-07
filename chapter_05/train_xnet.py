#-*-coding:utf-8-*-
# date:2019-05-20
# Author: X.L.Eric
# function: train P-Net/R-Net/O-Net

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from data_iter.mt_data_xnet_iter import *
from data_iter.utils_mt import *
import os
import numpy as np
from core.models import PNet,RNet,ONet,LossFn
import time
import datetime
import torch.nn as nn
from torch.autograd import Variable
import argparse
import multiprocessing

def mkdir_(path, flag_rm=False):
    if os.path.exists(path):
        if flag_rm == True:
            shutil.rmtree(path)
            os.mkdir(path)
            print('remove {} done ~ '.format(path))
    else:
        os.mkdir(path)

def trainer(ops):

    set_seed(ops.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    if ops.pattern == 'P-Net':
        m_XNet = PNet()
        mtcnn_detector = None
    elif ops.pattern == 'R-Net':
        m_XNet = RNet()
    elif ops.pattern == 'O-Net':
        m_XNet = ONet()
    # datasets
    dataset = LoadImagesAndLabels(pattern = ops.pattern,path_img = ops.path_img,path_anno = ops.path_anno,batch_size = ops.batch_size)
    print('dataset len : ',dataset.__len__())
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=ops.num_workers,
                            shuffle=True,
                            pin_memory=False,
                            drop_last = True)

    print('{} : \n'.format(ops.pattern),m_XNet)
    m_XNet = m_XNet.to(device)

    m_loss = LossFn()

    if ops.Optimizer_X == 'Adam':
        optimizer = torch.optim.Adam(m_XNet.parameters(), lr=ops.init_lr, betas=(0.9, 0.99),weight_decay=1e-6)
    elif ops.Optimizer_X == 'SGD':
        optimizer = torch.optim.SGD(m_XNet.parameters(), lr = ops.init_lr, momentum=0.9,weight_decay=1e-6)
    elif ops.Optimizer_X == 'RMSprop':
        optimizer=torch.optim.RMSprop(m_XNet.parameters(),lr=ops.init_lr,alpha=0.9,weight_decay=1e-6)
    else:
        print('------>>> Optimizer init error : ',ops.Optimizer_X)

    # load finetune model
    if os.access(ops.ft_model,os.F_OK):
        chkpt = torch.load(ops.ft_model, map_location=device)
        print('chkpt:\n',ops.ft_model)
        m_XNet.load_state_dict(chkpt)

    # train
    print('  epoch : ',ops.epochs)
    best_loss = np.inf
    loss_mean = 0.
    loss_cls_mean = 0.
    loss_idx = 0.
    init_lr = ops.init_lr
    
    loss_cnt = 0

    loss_cnt = 0


    for epoch in range(0, ops.epochs):

        if loss_idx!=0:
            if best_loss > (loss_mean/loss_idx):
                best_loss = loss_mean/loss_idx
                loss_cnt = 0
            else:
                if loss_cnt > 3:
                    init_lr = init_lr*0.5
                    set_learning_rate(optimizer, init_lr)
                    loss_cnt = 0
                else:
                    loss_cnt += 1

        loss_mean = 0.
        loss_cls_mean = 0.
        loss_idx = 0.

        print('\nepoch %d '%epoch)
        m_XNet = m_XNet.train()
        random.shuffle (dataset.annotations)# shuffle 图片组合

        for i, (imgs,gt_labels,gt_offsets,pos_num,part_num,neg_num) in enumerate(dataloader):
            imgs = imgs.squeeze(0)
            gt_labels = gt_labels.squeeze(0)
            gt_offsets = gt_offsets.squeeze(0)
            # print('imgs size {}, labels size {}, offsets size {}'.format(imgs.size(),gt_labels.size(),gt_offsets.size()))

            if use_cuda:
                imgs = imgs.cuda()  # (bs, 3, h, w)
                gt_labels = gt_labels.cuda()
                gt_offsets = gt_offsets.cuda()

            cls_pred, box_offset_pred = m_XNet(imgs)

            cls_loss = m_loss.focal_Loss(gt_labels,cls_pred)
            box_offset_loss = m_loss.box_loss(gt_labels,gt_offsets,box_offset_pred)

            if ops.pattern == 'O-Net':
                all_loss = cls_loss*1.0+box_offset_loss*0.4
            elif ops.pattern == 'R-Net':
                all_loss = cls_loss*1.0+box_offset_loss*0.6
            else:
                all_loss = cls_loss*1.0+box_offset_loss*1.0

            loss_mean += all_loss.item()
            loss_cls_mean += cls_loss.item()
            loss_idx += 1.

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

            if i%5 == 0:
                loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

                print("[%s]-<%s>Epoch: %d, [%d/%d],lr:%.6f,Loss:%.5f - Mean Loss:%.5f - Mean cls loss:%5f,cls_loss:%.5f ,bbox_loss:%.5f,imgs_batch: %4d,best_loss: %.5f" \
                % (loc_time,ops.pattern,epoch, i,dataset.__len__(), optimizer.param_groups[0]['lr'], \
                all_loss.item(),loss_mean/loss_idx,loss_cls_mean/loss_idx,cls_loss.item(),box_offset_loss.item(),imgs.size()[0],best_loss), ' ->pos:{},part:{},neg:{}'.format(pos_num.item(),part_num.item(),neg_num.item()))

            if i%50==0 and i>1:
                accuracy=compute_accuracy(cls_pred,gt_labels)
                print("\n  ------------- >>>  accuracy: %f\n"%(accuracy.item()))
                accuracy=compute_accuracy(cls_pred,gt_labels)
                torch.save(m_XNet.state_dict(), ops.ckpt + '{}_latest.pth'.format(ops.pattern))
            if i%80==0 and i>1:
                torch.save(m_XNet.state_dict(), ops.ckpt + '{}_epoch-{}.pth'.format(ops.pattern,epoch))

if __name__ == "__main__":

    # 当提示 GPU 不支持多进程，建议 ’spawn’，开启以下语句
    # multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description=' Project X-Net Train')
    parser.add_argument('--seed', type=int, default = 80,
        help = 'seed') # 设置随机种子
    parser.add_argument('--path_anno', type=str, default = "./datasets/wider_face_train.txt",
        help = 'path_anno')
    parser.add_argument('--path_img', type=str, default = "./datasets/WIDER_train/images/",
        help = 'path_img')
    parser.add_argument('--batch_size', type=int, default = 2048,
        help = 'batch_size')
    parser.add_argument('--init_lr', type=float, default = 1e-3,
        help = 'init_lr')
    parser.add_argument('--num_workers', type=int, default = 10,
        help = 'num_workers')
    parser.add_argument('--epochs', type=int, default = 1000,
        help = 'epochs')

    parser.add_argument('--ft_model', type=str, default = './ckpt/R-Net_latest.pth',#./ckpt/R-Net_latest.pth

        help = 'ft_model')
    parser.add_argument('--ckpt', type=str, default = './ckpt/',
        help = 'ckpt')
    parser.add_argument('--Optimizer_X', type=str, default = 'Adam',
        help = 'Optimizer_X：Adam,SGD,RMSprop')

    parser.add_argument('--pattern', type=str, default = 'R-Net',
        help = 'pattern：P-Net,R-Net,O-Net')

    #--------------------------------------------------------------------------
    args = parser.parse_args()# 解析添加参数

    mkdir_(args.ckpt)

    trainer(ops = args)
    print('well done : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
