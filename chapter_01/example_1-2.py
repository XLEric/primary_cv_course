#-*-coding:utf-8-*-
# date:2020-03-28
# Author: X.L.Eric
# function: image pixel - uint8 (0~255)

import cv2 # 加载 OpenCV 库
import numpy as np # 加载 numpy 库
if __name__ == "__main__":
    img_h = 480
    img_w = 640
    img = np.zeros([img_h,img_w], dtype = np.uint8)

    cv2.namedWindow('image_0', 1)
    cv2.imshow('image_0', img)
    cv2.waitKey(0)

    img.fill(100)
    cv2.namedWindow('image_100', 1)
    cv2.imshow('image_100', img)
    cv2.waitKey(0)

    img.fill(255)
    cv2.namedWindow('image_255', 1)
    cv2.imshow('image_255', img)
    cv2.waitKey(0)
