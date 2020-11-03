#-*-coding:utf-8-*-
# date:2019-05-20
# Author: X.L.Eric
# function: video inference detect face & landmarks

import os
import numpy as np
from core.models import PNet,RNet,ONet
import time
import datetime
import torch.nn as nn
from core.detect import *

from models.resnet_50 import resnet50

from utils import get_faces_batch_landmarks

if __name__ == "__main__":
    Camera_Width = 640 #摄像头宽
    Camera_High  = 480 #摄像头高
    video_capture = cv2.VideoCapture('./video/rw_7.mp4')
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,Camera_Width)  #设置摄像头宽
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,Camera_High)  #设置摄像头高

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print('device:',device)

    p_model_path = './ckpt/P-Net_latest.pth'
    r_model_path = './ckpt/R-Net_latest.pth'
    o_model_path = './ckpt/O-Net_latest.pth'

    m_PNet,m_RNet,m_ONet = create_XNet(device,p_model_path,r_model_path,o_model_path)

    mtcnn_detector = MtcnnDetector(pnet=m_PNet,rnet=m_RNet,onet=m_ONet,
        min_face_size=60,threshold=[0.45, 0.5, 0.75])
    #
    landmarks_model=resnet50(num_classes=196)
    path_landmarks = './ckpt/landmarks.pth'
    if os.access(path_landmarks,os.F_OK):# checkpoint
        ckpt = torch.load(path_landmarks, map_location=device)
        landmarks_model.load_state_dict(ckpt)
        print('load landmarks model : {}'.format(path_landmarks))
    else:
        print('error load landmarks model ')

    landmarks_model = landmarks_model.to(device)
    landmarks_model.eval() # 设置为前向推断模式
    #---------------------------------------------------------------------------
    print('init MtcnnDetector')
    got_fps = 0.
    t_tk_0 = time.time()
    show_fps_cnt = 0.
    stop_flag = False

    while True:
        ret, frame_n = video_capture.read()
        # print('----------------------->>> ',frame_n.shape)

        t_delta_tk = time.time() - t_tk_0
        if (t_delta_tk)>=1.0:
            show_fps_cnt = float(got_fps)/ t_delta_tk
            got_fps =0.
            t_tk_0 = time.time()
        got_fps += 1
        if ret:
            frame_r = frame_n.copy()
            with torch.no_grad():
                boxes_align, _ = mtcnn_detector.detect_face(img=frame_r )

            if boxes_align is not None:
                for i in range(len(boxes_align)):
                    box = boxes_align[i]
                    x_min = int(box[0])
                    y_min = int(box[1])
                    x_max = int(box[2])
                    y_max = int(box[3])
                    thr = box[4]
                    face_h = y_max-y_min
                    face_w = x_max-x_min
                    # print('----------->>> right ',box[4],x_min,y_min,x_max,y_max)
                    cv2.rectangle(frame_n, (x_min,y_min), (x_max,y_max), (0,255,225), 2)
                    cv2.putText(frame_n, ("%2f-(%3d,%3d)" %(thr,face_h,face_w)), (x_min,y_min),\
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 55, 55), 2)

                get_faces_batch_landmarks(landmarks_model,boxes_align,frame_n,use_cuda,draw_bbox = True)


            cv2.putText(frame_n, (" fps :%f" % show_fps_cnt), (10,frame_n.shape[0]-5),\
            cv2.FONT_HERSHEY_PLAIN, 1.1, (55, 155, 255), 3)
            cv2.putText(frame_n, (" fps :%f" % show_fps_cnt),(10,frame_n.shape[0]-5),\
            cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 155, 155), 1)
            cv2.namedWindow('image',0)
            cv2.imshow('image',frame_n)

        key_id = cv2.waitKey(1)

        if key_id == 27:
            break
        if key_id == ord('s') or key_id == ord('S'):
            stop_flag = True
        while stop_flag:
            key_id2 = cv2.waitKey(80)
            if key_id2 == ord('s') or key_id2 == ord('S'):
                stop_flag = False

    video_capture.release()
    cv2.destroyAllWindows()

    print('well done ~')
