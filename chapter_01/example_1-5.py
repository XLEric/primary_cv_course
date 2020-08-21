#-*-coding:utf-8-*-
# date:2020-03-30
# Author: X.L.Eric
# function: image [B,G,R] split & [B,G,R] merge 
import cv2 # 加载OpenCV库
import numpy as np
if __name__ == "__main__":
    img = cv2.imread('./samples/42_Car_Racing_Car_Racing_42_857.jpg')# 读取图片
    cv2.namedWindow('image',0)
    cv2.imshow('image',img) # 显示图片

    #图片3通道分离
    (B,G,R) = cv2.split(img)
    cv2.namedWindow('B', 0)
    cv2.imshow('B', B)
    cv2.namedWindow('G', 0)
    cv2.imshow('G', G)
    cv2.namedWindow('R', 0)
    cv2.imshow('R', R)

    # 图片合并
    img_merge = cv2.merge([B,G,R])

    cv2.namedWindow('merge', 0)
    cv2.imshow('merge', img_merge)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
