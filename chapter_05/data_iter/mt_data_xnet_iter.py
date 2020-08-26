#coding:utf-8
#-*-coding:utf-8-*-
# date:2019-07-20
# Author: X.L.Eric
# function: data iter
import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from data_iter.utils_mt import IoU
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compute_accuracy(prob_cls, gt_cls):

    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)

    #we only need the detection which >= 0
    mask = torch.ge(gt_cls,0)
    #get valid element
    valid_gt_cls = torch.masked_select(gt_cls,mask)
    valid_prob_cls = torch.masked_select(prob_cls,mask)
    size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
    prob_ones = torch.ge(valid_prob_cls,0.6).float()
    right_ones = torch.eq(prob_ones,valid_gt_cls).float()

    ## if size == 0 meaning that your gt_labels are all negative, landmark or part

    return torch.div(torch.mul(torch.sum(right_ones),float(1.0)),float(size))  ## divided by zero meaning that your gt_labels are all negative, landmark or part

def contrast_img(img, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img.shape
    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img.dtype)
    dst = cv2.addWeighted(img, c, blank, 1-c, b)
    return dst

def img_agu(img):
    if random.random()>0.5:
        c = float(random.randint(50,150))/100.
        b = random.randint(-20,20)
        img = contrast_img(img, c, b)
        # print('------------->>> contrast_img')
    if random.random()>0.85:#颜色通道变换
        (B,G,R) = cv2.split(img)

        id_C = random.randint(0,6)
        # print('idc',id_C)
        if id_C ==0:
            img = cv2.merge([B,G,R])
        elif id_C ==1:
            img = cv2.merge([B,R,G])
        elif id_C ==2:
            img = cv2.merge([R,G,B])
        elif id_C ==3:
            img = cv2.merge([R,B,G])
        elif id_C ==4:
            img = cv2.merge([G,B,R])
        elif id_C ==5:
            img = cv2.merge([G,R,B])
        elif id_C ==6:
            gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            img = cv2.merge([gray,gray,gray])
        # print('------------->>> color')

    if random.random()>0.96:
        img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hue_x = random.randint(-10,10)
        # print(cc)
        img_hsv[:,:,0]=(img_hsv[:,:,0]+hue_x)
        img_hsv[:,:,0] =np.maximum(img_hsv[:,:,0],0)
        img_hsv[:,:,0] =np.minimum(img_hsv[:,:,0],180)#范围 0 ~180
        img=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
        # print('------------->>> hsv')
    return img

class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, pattern,path_img,path_anno,batch_size,flag_debug = False):

        if pattern == 'P-Net':
            img_size=12
            img_min_size = 24
        elif pattern == 'R-Net':
            img_size=24
            img_min_size = 36
        elif pattern == 'O-Net':
            img_size=48
            img_min_size = 60

        self.pattern = pattern
        self.img_size = img_size
        self.img_min_size = img_min_size

        with open(path_anno, 'r') as f:
            annotations = f.readlines()
        num = len(annotations)
        print(num,"pics in total")

        datas_index_list = list(range(len(annotations)))

        self.path_img = path_img
        self.batch_size = batch_size
        self.datas_index_list = datas_index_list
        self.pics_num = num
        self.annotations = annotations
        self.flag_debug = flag_debug
        # random.shuffle (self.datas_index_list)
        random.shuffle (self.annotations)

    def __len__(self):
        return (len(self.annotations)-2048)

    def __getitem__(self, index):
        imgs_list = []
        offsets_list = []
        labels_list  = []

        if self.flag_debug : print("{}~{}".format(index,min(self.pics_num,index+self.batch_size)))
        #--------------------------
        batch_idx_list = list(range(index,min(self.pics_num,index+self.batch_size)))

        random.shuffle(batch_idx_list)
        batch_idx_list_pos = batch_idx_list[0:max(1,int(len(batch_idx_list)*2/9))]

        random.shuffle(batch_idx_list)
        batch_idx_list_part = batch_idx_list[0:max(1,int(len(batch_idx_list)*3/9))]

        random.shuffle(batch_idx_list)
        batch_idx_list_neg = batch_idx_list[0:max(1,int(len(batch_idx_list)*4/9))]
        #--------------------------
        pos_num_sum = 0
        part_num_sum = 0
        neg_num_sum = 0

        pos_num_sum_thr = 100 + random.randint(-20,20)
        part_num_sum_thr = 200 + random.randint(-30,30)
        neg_num_sum_thr = 300 + random.randint(-50,50)



        # for jj in range(index,min(self.pics_num,index+self.batch_size)):
        #---------------------------------------------
        random.shuffle(batch_idx_list)
        for kk in range(len(batch_idx_list)):
            jj = batch_idx_list[kk]
        #---------------------------------------------
            if (pos_num_sum>= pos_num_sum_thr) and (part_num_sum>= part_num_sum_thr) and (neg_num_sum>= neg_num_sum_thr):
                break

            if not((jj in batch_idx_list_pos) or (jj in batch_idx_list_part) or (jj in batch_idx_list_neg)):
                continue
            annotation = self.annotations[jj].strip().split(' ')
            #image path
            im_path = annotation[0]

            bbox = list(map(float, annotation[1:]))
            #gt
            boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
            if self.flag_debug:
                print('   {} {} '.format(jj,im_path))
            #boxed change to float type
            bbox = list(map(float, annotation[1:]))
            #gt
            boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
            #load image
            read_img_path = os.path.join(self.path_img, im_path + '.jpg')

            img = cv2.imread(read_img_path)

            height, width, channel = img.shape
            #------------------------------------------ gt box
            if self.flag_debug:
                img_c = img.copy()
                for box in boxes:
                    # box (x_left, y_top, x_right, y_bottom)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(img_c,(x1,y1), (x2,y2), (255,90,90), 2)

                cv2.namedWindow('img',0)
                cv2.imshow('img',img_c)
                cv2.waitKey(1)
            #------------------------------------------------------------------- pos,part,neg 数量阈值
            # pos,part,neg label : 1.0,-1.0,0.0

            #------------------------------------------ pos box
            pos_num = 0
            #1---->50
            time_out  = 0
            random.shuffle (boxes)

            boxes_idx_list = random.shuffle (list(range(len(boxes))))

            while (pos_num < 1) and  (pos_num_sum< pos_num_sum_thr) and (jj in batch_idx_list_pos):
                # random.randint
                if time_out>(len(boxes)*100):
                    break

                for box in boxes:

                    # box (x_left, y_top, x_right, y_bottom)
                    x1, y1, x2, y2 = box
                    #gt's width
                    w = x2 - x1 + 1
                    #gt's height
                    h = y2 - y1 + 1
                    # ignore small faces
                    # in case the ground truth boxes of small faces are not accurate
                    time_out += 1
                    # print('time_out : ',time_out)
                    if min(w, h) < self.img_min_size or x1 < 0 or y1 < 0:
                        continue

                    # cv2.rectangle(img,(x1,y1), (x2,y2), (255,90,90), 2)
                    #----------------------------------------------------------------------------- pos
                    # pos and part face size [minsize*0.8,maxsize*1.25]
                    size = npr.randint(int(min(w, h) * 0.75), np.ceil(1.25 * max(w, h)))

                    # delta here is the offset of box center
                    delta_x = npr.randint(-w * 0.2, w * 0.2)
                    delta_y = npr.randint(-h * 0.2, h * 0.2)
                    #show this way: nx1 = max(x1+w/2-size/2+delta_x)
                    nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                    #show this way: ny1 = max(y1+h/2-size/2+delta_y)
                    ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                    nx2 = nx1 + size
                    ny2 = ny1 + size

                    if nx2 > width or ny2 > height:
                        continue
                    # print('AAAAAAAAAAAAAAAAAAAAA')
                    crop_box = np.array([nx1, ny1, nx2, ny2])
                    #yu gt de offset
                    offset_x1 = (x1 - nx1) / float(size)
                    offset_y1 = (y1 - ny1) / float(size)
                    offset_x2 = (x2 - nx2) / float(size)
                    offset_y2 = (y2 - ny2) / float(size)


                    box_ = box.reshape(1, -1)
                    if IoU(crop_box, box_) >= 0.65:
                        #crop
                        cropped_im = img[ny1 : ny2, nx1 : nx2, :]
                        #resize
                        resized_im = cv2.resize(cropped_im, (self.img_size, self.img_size), interpolation=random.randint(0,4))

                        resized_im = img_agu(resized_im)
                        if random.random()>0.5:#水平翻转  千万不能做 水平翻转对应的 offset 一定要变
                            img_ = cv2.flip(resized_im,1)
                            offset_x1,offset_x2 = -offset_x2,-offset_x1


                        img_ = resized_im.transpose(2, 0, 1)
                        #--------------
                        imgs_list.append(img_)
                        offsets_list.append((offset_x1, offset_y1, offset_x2, offset_y2))
                        labels_list.append(1.)
                        #--------------
                        # print('cropped_im.shape',cropped_im.shape,offset_x1, offset_y1, offset_x2, offset_y2)

                        if self.flag_debug:
                            cv2.namedWindow('pos',0)
                            cv2.imshow('pos',cropped_im)
                            cv2.waitKey(1)
                        pos_num += 1
                        break


            if self.flag_debug : print('    -------->>>> pos : ',pos_num)

            #-------------------------------------------------------------- part
            part_num = 0
            time_out  = 0
            random.shuffle (boxes)
            #1---->50
            while (part_num <1) and (part_num_sum< part_num_sum_thr) and (jj in batch_idx_list_part):
                # random.randint
                if time_out>(len(boxes)*120):
                    break
                for box in boxes:
                    # box (x_left, y_top, x_right, y_bottom)
                    x1, y1, x2, y2 = box
                    #gt's width
                    w = x2 - x1 + 1
                    #gt's height
                    h = y2 - y1 + 1
                    # ignore small faces
                    # in case the ground truth boxes of small faces are not accurate
                    time_out += 1

                    if min(w, h) < self.img_min_size or x1 < 0 or y1 < 0:
                        continue

                    # cv2.rectangle(img,(x1,y1), (x2,y2), (255,90,90), 2)
                    #-----------------------------------------------------------------------------
                    # pos and part face size [minsize*0.8,maxsize*1.25]
                    size = npr.randint(int(min(w, h) * 0.75), np.ceil(1.25 * max(w, h)))

                    # delta here is the offset of box center
                    delta_x = npr.randint(-w * 0.2, w * 0.2)
                    delta_y = npr.randint(-h * 0.2, h * 0.2)
                    #show this way: nx1 = max(x1+w/2-size/2+delta_x)
                    nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                    #show this way: ny1 = max(y1+h/2-size/2+delta_y)
                    ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                    nx2 = nx1 + size
                    ny2 = ny1 + size

                    if nx2 > width or ny2 > height:
                        continue
                    crop_box = np.array([nx1, ny1, nx2, ny2])
                    #yu gt de offset
                    offset_x1 = (x1 - nx1) / float(size)
                    offset_y1 = (y1 - ny1) / float(size)
                    offset_x2 = (x2 - nx2) / float(size)
                    offset_y2 = (y2 - ny2) / float(size)

                    box_ = box.reshape(1, -1)
                    if 0.65 > IoU(crop_box, box_) >= 0.45:
                        #crop
                        cropped_im = img[ny1 : ny2, nx1 : nx2, :]
                        #resize
                        resized_im = cv2.resize(cropped_im, (self.img_size, self.img_size), interpolation=random.randint(0,4))
                        #

                        resized_im = img_agu(resized_im)
                        if random.random()>0.5:#水平翻转  千万不能做 水平翻转对应的 offset 一定要变
                            img_ = cv2.flip(resized_im,1)
                            offset_x1,offset_x2 = -offset_x2,-offset_x1
                            img_ = img_.transpose(2, 0, 1)
                        else:
                            img_ = resized_im.transpose(2, 0, 1)
                        #--------------
                        imgs_list.append(img_)
                        offsets_list.append((offset_x1, offset_y1, offset_x2, offset_y2))
                        labels_list.append(-1.)
                        #--------------
                        if self.flag_debug:
                            cv2.namedWindow('part',0)
                            cv2.imshow('part',cropped_im)
                            cv2.waitKey(1)
                        part_num += 1
                        break

            if self.flag_debug : print('    -------->>>> part : ',part_num)

            #------------------------------------------ neg

            neg_num = 0
            #1---->50

            while (neg_num < 1) and (neg_num_sum< neg_num_sum_thr) and (jj in batch_idx_list_neg):

                #neg_num's size [40,min(width, height) / 2],min_size:40
                size = npr.randint(12, min(width, height) / 2)
                #top_left
                nx = npr.randint(0, width - size)# 做的样本是一个边长相等的 正方形框
                ny = npr.randint(0, height - size)
                #random crop
                crop_box = np.array([nx, ny, nx + size, ny + size])
                #cal iou
                Iou = IoU(crop_box, boxes)

                if np.max(Iou) < 0.3:
                    # Iou with all gts must below 0.3
                    cropped_im = img[ny : ny + size, nx : nx + size, :]
                    resized_im = cv2.resize(cropped_im, (self.img_size, self.img_size), interpolation=random.randint(0,4))
                    #

                    resized_im = img_agu(resized_im)

                    if random.random()>0.5:#水平翻转
                        img_ = cv2.flip(resized_im,1)
                        img_ = img_.transpose(2, 0, 1)
                    else:
                        img_ = resized_im.transpose(2, 0, 1)
                    #--------------
                    imgs_list.append(img_)
                    offsets_list.append((0,0,0,0))
                    labels_list.append(0.0)
                    #-------------
                    if self.flag_debug:
                        cv2.namedWindow('neg',0)
                        cv2.imshow('neg',cropped_im)
                        cv2.waitKey(1)
                    neg_num += 1

            #------------------------------------------- pos
            if self.flag_debug : print('    -------->>>> neg : ',neg_num)

            pos_num_sum += pos_num
            part_num_sum += part_num
            neg_num_sum += neg_num

        if self.flag_debug : print('imgs_list len {} , offsets_list len {} , labels_list len {} '.format(len(imgs_list),len(offsets_list),len(labels_list)))
        imgs_list = np.array(imgs_list)
        imgs = imgs_list.astype(np.float32)

        offsets_list = np.array(offsets_list)
        gt_offsets = offsets_list.astype(np.float32)

        labels_list = np.array(labels_list)
        gt_labels = labels_list.astype(np.float32)

        if self.flag_debug : print('imgs_list shape {} , gt_offsets shape {} , gt_labels shape {} '.format(imgs.shape,gt_offsets.shape,gt_labels.shape))

        return imgs,gt_labels,gt_offsets,pos_num_sum,part_num_sum,neg_num_sum




def data_iter_fun(path_img,path_anno,flag_debug = False):
    with open(path_anno, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print(num,"pics in total")

    datas_index_list = list(range(len(annotations)))
    # random.shuffle(datas_index_list)
    # random.shuffle (annotations)

    # print('datas_index_list : ',datas_index_list)
    batch_size = 8


    p_idx = 0 # positive
    n_idx = 0 # negative
    d_idx = 0 # dont care


    for ii in range(len(datas_index_list)):
        print('------>>> {} --->>> index range: {}~{}'.format(ii,datas_index_list[ii],datas_index_list[ii]+batch_size))
        for jj in range(datas_index_list[ii],min(num,datas_index_list[ii]+batch_size)):
            annotation = annotations[jj].strip().split(' ')
            #image path
            im_path = annotation[0]

            bbox = list(map(float, annotation[1:]))
            #gt
            boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)

            print('   {} {} '.format(jj,im_path))
            #boxed change to float type
            bbox = list(map(float, annotation[1:]))
            #gt
            boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
            #load image
            read_img_path = os.path.join(path_img, im_path + '.jpg')

            img = cv2.imread(read_img_path)

            height, width, channel = img.shape
            #------------------------------------------ gt box
            img_c = img.copy()
            for box in boxes:
                # box (x_left, y_top, x_right, y_bottom)
                x1, y1, x2, y2 = box
                cv2.rectangle(img_c,(x1,y1), (x2,y2), (255,90,90), 2)

            cv2.namedWindow('img',0)
            cv2.imshow('img',img_c)
            cv2.waitKey(1)
            #------------------------------------------
            pos_num_thr = max(1,min(10,len(boxes)))
            part_num_thr = 2*pos_num_thr
            neg_num_thr = 3*pos_num_thr

            #------------------------------------------ pos box

            pos_num = 0
            #1---->50
            time_out  = 0
            random.shuffle (boxes)
            while pos_num < pos_num_thr:
                # random.randint
                if time_out>(len(boxes)*100):
                    break

                for box in boxes:
                    # box (x_left, y_top, x_right, y_bottom)
                    x1, y1, x2, y2 = box
                    #gt's width
                    w = x2 - x1 + 1
                    #gt's height
                    h = y2 - y1 + 1
                    # ignore small faces
                    # in case the ground truth boxes of small faces are not accurate
                    time_out += 1
                    # print('time_out : ',time_out)
                    if min(w, h) < 12 or x1 < 0 or y1 < 0:
                        continue

                    # cv2.rectangle(img,(x1,y1), (x2,y2), (255,90,90), 2)
                    #----------------------------------------------------------------------------- pos
                    # pos and part face size [minsize*0.8,maxsize*1.25]
                    size = npr.randint(int(min(w, h) * 0.75), np.ceil(1.25 * max(w, h)))

                    # delta here is the offset of box center
                    delta_x = npr.randint(-w * 0.2, w * 0.2)
                    delta_y = npr.randint(-h * 0.2, h * 0.2)
                    #show this way: nx1 = max(x1+w/2-size/2+delta_x)
                    nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                    #show this way: ny1 = max(y1+h/2-size/2+delta_y)
                    ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                    nx2 = nx1 + size
                    ny2 = ny1 + size

                    if nx2 > width or ny2 > height:
                        continue
                    # print('AAAAAAAAAAAAAAAAAAAAA')
                    crop_box = np.array([nx1, ny1, nx2, ny2])
                    #yu gt de offset
                    offset_x1 = (x1 - nx1) / float(size)
                    offset_y1 = (y1 - ny1) / float(size)
                    offset_x2 = (x2 - nx2) / float(size)
                    offset_y2 = (y2 - ny2) / float(size)
                    #crop
                    cropped_im = img[ny1 : ny2, nx1 : nx2, :]
                    #resize
                    resized_im = cv2.resize(cropped_im, (12, 12), interpolation=random.randint(0,4))

                    box_ = box.reshape(1, -1)
                    if IoU(crop_box, box_) >= 0.65:
                        # print('BBBBBBBBBBBBBBBBBBBB')
                        # save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                        # f1.write("Pnet_12/positive/%s.jpg"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                        # cv2.imwrite(save_file, resized_im)
                        p_idx += 1
                        cv2.namedWindow('pos',0)
                        cv2.imshow('pos',cropped_im)
                        cv2.waitKey(1)
                        pos_num += 1
                        if pos_num >= pos_num_thr:
                            break

            print('    -------->>>> pos : ',pos_num)

            #-------------------------------------------------------------- part
            part_num = 0
            time_out  = 0
            random.shuffle (boxes)
            #1---->50
            while part_num < part_num_thr:
                # random.randint
                if time_out>(len(boxes)*100):
                    break
                for box in boxes:
                    # box (x_left, y_top, x_right, y_bottom)
                    x1, y1, x2, y2 = box
                    #gt's width
                    w = x2 - x1 + 1
                    #gt's height
                    h = y2 - y1 + 1
                    # ignore small faces
                    # in case the ground truth boxes of small faces are not accurate
                    time_out += 1

                    if min(w, h) < 12 or x1 < 0 or y1 < 0:
                        continue

                    # cv2.rectangle(img,(x1,y1), (x2,y2), (255,90,90), 2)
                    #-----------------------------------------------------------------------------
                    # pos and part face size [minsize*0.8,maxsize*1.25]
                    size = npr.randint(int(min(w, h) * 0.75), np.ceil(1.25 * max(w, h)))

                    # delta here is the offset of box center
                    delta_x = npr.randint(-w * 0.2, w * 0.2)
                    delta_y = npr.randint(-h * 0.2, h * 0.2)
                    #show this way: nx1 = max(x1+w/2-size/2+delta_x)
                    nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                    #show this way: ny1 = max(y1+h/2-size/2+delta_y)
                    ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                    nx2 = nx1 + size
                    ny2 = ny1 + size

                    if nx2 > width or ny2 > height:
                        continue
                    crop_box = np.array([nx1, ny1, nx2, ny2])
                    #yu gt de offset
                    offset_x1 = (x1 - nx1) / float(size)
                    offset_y1 = (y1 - ny1) / float(size)
                    offset_x2 = (x2 - nx2) / float(size)
                    offset_y2 = (y2 - ny2) / float(size)
                    #crop
                    cropped_im = img[ny1 : ny2, nx1 : nx2, :]
                    #resize
                    resized_im = cv2.resize(cropped_im, (12, 12), interpolation=random.randint(0,4))

                    box_ = box.reshape(1, -1)
                    if 0.65 > IoU(crop_box, box_) >= 0.40:
                        # save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                        # f3.write("Pnet_12/part/%s.jpg"%d_idx + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                        # cv2.imwrite(save_file, resized_im)
                        d_idx += 1
                        cv2.namedWindow('part',0)
                        cv2.imshow('part',cropped_im)
                        cv2.waitKey(1)
                        part_num += 1
                        if part_num >= part_num_thr:
                            break
            print('    -------->>>> part : ',part_num)

            #------------------------------------------ neg

            neg_num = 0
            #1---->50
            while neg_num < max((len(boxes)*3),neg_num_thr):
                #neg_num's size [40,min(width, height) / 2],min_size:40
                size = npr.randint(12, min(width, height) / 2)
                #top_left
                nx = npr.randint(0, width - size)# 做的样本是一个边长相等的 正方形框
                ny = npr.randint(0, height - size)
                #random crop
                crop_box = np.array([nx, ny, nx + size, ny + size])
                #cal iou
                Iou = IoU(crop_box, boxes)

                cropped_im = img[ny : ny + size, nx : nx + size, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                if np.max(Iou) < 0.3:
                    # Iou with all gts must below 0.3
                    # save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
                    # f2.write("Pnet_12/negative/%s.jpg"%n_idx + ' 0\n')
                    # cv2.imwrite(save_file, resized_im)
                    n_idx += 1
                    cv2.namedWindow('neg',0)
                    cv2.imshow('neg',cropped_im)
                    cv2.waitKey(1)
                    neg_num += 1
            #------------------------------------------- pos
            print('    -------->>>> neg : ',neg_num)


            if cv2.waitKey(1) == 27:
                break

if __name__ == '__main__':

    path_anno = "../datasets/wider_face_train.txt"
    path_img = "../datasets/WIDER_train/images/"

    data_iter_fun(path_img = path_img,path_anno = path_anno,flag_debug = False)
