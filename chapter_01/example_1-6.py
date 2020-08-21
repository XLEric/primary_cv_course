#-*-coding:utf-8-*-
# date:2020-03-28
# Author: X.L
# function: flip image

import cv2 #导入OpenCV数据库
if __name__ == "__main__":
    img = cv2.imread('./samples/42_Car_Racing_Car_Racing_42_857.jpg')# 读取图片
    cv2.namedWindow('image',0)
    cv2.imshow('image',img) # 显示图片

    img_flip = cv2.flip(img,1) # 图像左右翻转
    cv2.namedWindow('img_flip',0)
    cv2.imshow('img_flip',img_flip) # 显示图片
    cv2.waitKey(0)# 等待按键按下

    img_flip = cv2.flip(img,0) # 图像上下翻转
    cv2.imshow('img_flip',img_flip) # 显示图片
    cv2.waitKey(0)# 等待按键按下

    img_flip = cv2.flip(img,-1) # 图像顺直水平同时翻转
    cv2.imshow('img_flip',img_flip) # 显示图片
    cv2.waitKey(0)# 等待按键按下

    cv2.destroyAllWindows()# 销毁图片显示窗口
