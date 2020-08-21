#-*-coding:utf-8-*-
# date:2020-03-28
# Author: X.L.Eric
# function: image pixel - float (0.~1.)

import cv2 # 加载 OpenCV 库
import numpy as np # 加载 numpy 库
if __name__ == "__main__":
    img_h = 480
    img_w = 640
    img = np.zeros([img_h,img_w], dtype = np.float)

    cv2.namedWindow('image_0', 1)
    cv2.imshow('image_0', img)
    cv2.waitKey(0)

    img.fill(0.5)
    cv2.namedWindow('image_05', 1)
    cv2.imshow('image_05', img)
    cv2.waitKey(0)

    img.fill(1.)
    cv2.namedWindow('image_1', 1)
    cv2.imshow('image_1', img)
    cv2.waitKey(0)
