#-*-coding:utf-8-*-
# date:2020-03-28
# Author: X.L.Eric
# function: image B G R show

import cv2 # 加载OpenCV库
import numpy as np
if __name__ == "__main__":
    img_h = 320
    img_w = 480

    img = np.zeros([img_h,img_w,3], dtype = np.float)
    img[:,:,0].fill(1.0)
    cv2.namedWindow('B', 1)
    cv2.imshow('B', img)

    img = np.zeros([img_h,img_w,3], dtype = np.float)
    img[:,:,1].fill(1.0)
    cv2.namedWindow('G', 1)
    cv2.imshow('G', img)

    img = np.zeros([img_h,img_w,3], dtype = np.float)
    img[:,:,2].fill(1.0)
    cv2.namedWindow('R', 1)
    cv2.imshow('R', img)

    cv2.waitKey(0)

    cv2.destroyAllWindows() # 销毁所有显示窗口
