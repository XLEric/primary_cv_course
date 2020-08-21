#-*-coding:utf-8-*-
# date:2020-03-28
# Author: X.L
# function: read image

import os
import cv2 # 加载OpenCV库
if __name__ == "__main__":
    path = './samples/'
    for file in os.listdir(path):
        img = cv2.imread(path+file)
        cv2.namedWindow('image',0)# 定义显示窗口
        cv2.imshow('image',img)# 显示图片

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)# BGR图像转灰度图像
        cv2.namedWindow('gray',0)# 定义显示窗口
        cv2.imshow('gray',gray) # 显示图片

        cv2.waitKey(0)
    cv2.destroyAllWindows()
