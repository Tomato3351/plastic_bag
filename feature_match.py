# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 09:59:54 2019

@author: TOMATO
"""

import cv2
import numpy as np
#import glob as gb
import matplotlib.pyplot as plt


def area_img(img,point,r=5):
    #计算某一点的邻域灰度值,输入灰度图像，待求点坐标,邻域块半径
    h,w=img.shape
    x,y=point
    row_start=max(0,y-r)
    row_end=min(y+r,h)
    col_start=max(0,x-r)
    col_end=min(x+r,w)
    area_img=img[row_start:row_end,col_start:col_end]
#    area_gray=np.mean(area)
    return area_img
    
def good_match(img1,img2):
    #基于orb和感知哈希算法计算匹配点。输入两幅图像，返回匹配点对,包括良好匹配点对和原始匹配点对。
#    h,w,c=img1.shape
    img1_gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2_gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    orb=cv2.ORB_create()    
    keyPoints1,descriptors1=orb.detectAndCompute(img1,None)
    keyPoints2,descriptors2=orb.detectAndCompute(img2,None)
    bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    good_matches=[]
    if len(keyPoints1)>0 and len(keyPoints2)>0:
        matches=bf.match(descriptors1,descriptors2)
        matches=sorted(matches,key=lambda x:x.distance)   
        
        for i in matches[:int(len(matches)/2)]:
            x1,y1=keyPoints1[i.queryIdx].pt#此点在img_left中的坐标
            x2,y2=keyPoints2[i.trainIdx].pt#此点在img_right中的坐标
            area_img1=area_img(img1_gray,(int(x1),int(y1)),64).copy()
            area_img2=area_img(img2_gray,(int(x2),int(y2)),64).copy()
            #感知哈希算法，计算两幅图像的相似度。hamming距离为5以内即为相似度满足要求。
            #进行dct变换
            dct_area1=cv2.dct(np.float32(cv2.resize(area_img1,(32,32))))
            dct_area1=dct_area1[0:8,0:8]#只保留系数矩阵左上角8*8的低频部分
            dct_area1_binary=(dct_area1>np.mean(dct_area1))
            dct_area2=cv2.dct(np.float32(cv2.resize(area_img2,(32,32))))
            dct_area2=dct_area2[0:8,0:8]#只保留系数矩阵左上角8*8的低频部分
            dct_area2_binary=(dct_area2>np.mean(dct_area2))
            hamming_dis=np.sum(dct_area1_binary^dct_area2_binary)    
            if i.distance<60 and hamming_dis<9:
                good_matches.append(i)
    else:
        print('no keyPoints!')
        matches=[]
    return good_matches,matches,keyPoints1,keyPoints2   
            
if __name__=="__main__":
#    img_paths=gb.glob("./imgs\\*.jpg")
#    img_paths=gb.glob("D:/MVS_DATA\\*.jpg")    
#    img=cv2.imread(img_paths[2],1)
#    img_left=cv2.imread("d:/python_projects/plastic_bag/imgs/55Left.jpg",-1)
#    img_right=cv2.imread("d:/python_projects/plastic_bag/imgs/55Right.jpg",-1)
    img_left=cv2.imread("d:/python_projects/plastic_bag/img_left.jpg",-1)
    img_right=cv2.imread("d:/python_projects/plastic_bag/img_right.jpg",-1)   
    
    h,w,c=img_left.shape
#    img_left=cv2.GaussianBlur(img_left,(5,5),0)#参数为高斯矩阵尺寸,标准偏差
#    img_right=cv2.GaussianBlur(img_right,(5,5),0)#参数为高斯矩阵尺寸,标准偏差
    cv2.namedWindow("img_left",0)
    cv2.imshow("img_left",img_left)
    cv2.waitKey(0)
    cv2.namedWindow("img_right",0)
    cv2.imshow("img_right",img_right)
    cv2.waitKey(0)    
    img_left_gray=cv2.cvtColor(img_left,cv2.COLOR_BGR2GRAY)
    img_right_gray=cv2.cvtColor(img_right,cv2.COLOR_BGR2GRAY)
    orb=cv2.ORB_create()#创建orb对象
    #寻找特征点并计算特征描述向量
    keyPoints1,descriptors1=orb.detectAndCompute(img_left,None)
    keyPoints2,descriptors2=orb.detectAndCompute(img_right,None)
    #BF匹配
    bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches=bf.match(descriptors1,descriptors2)
    matches=sorted(matches,key=lambda x:x.distance)
    #画出匹配点
    imgMatch=cv2.drawMatches(img_left,keyPoints1,img_right,
                         keyPoints2,matches,None,flags=2)
    imgMatch_rgb=cv2.cvtColor(imgMatch,cv2.COLOR_BGR2RGB)
    plt.figure("imgMatch",figsize=(18,6))      
    plt.imshow(imgMatch_rgb)#cmap='gray'   
    good_matches=[]
#    cv2.namedWindow('local_img1',0)
#    cv2.namedWindow('local_img2',0)
    plt.figure("local_imgs",figsize=(16,8))
    count=0
    for i in matches[:int(len(matches)/2)]:
        print(count,' distance=',i.distance)
        #kp1[0].pt第一个关键点位置坐标
        #kp1[0].size关键点邻域直径
        #DMatch.distance - 描述符之间的距离。越小越好。
        #DMatch.trainIdx - 目标图像中描述符的索引。
        #DMatch.queryIdx - 查询图像中描述符的索引。
        #DMatch.imgIdx - 目标图像的索引。 
        x1,y1=keyPoints1[i.queryIdx].pt#此点在img_left中的坐标
        x2,y2=keyPoints2[i.trainIdx].pt#此点在img_right中的坐标
#        print(x1,y1,'\n',x2,y2)
#        print(x1-x2,y1-y2)
        #特征点局部图像
        area_img1=area_img(img_left_gray,(int(x1),int(y1)),64).copy()
        area_img2=area_img(img_right_gray,(int(x2),int(y2)),64).copy()
        #感知哈希算法，计算两幅图像的相似度。hamming距离为5以内即为相似度满足要求。
        #进行dct变换
        dct_area1=cv2.dct(np.float32(cv2.resize(area_img1,(32,32))))
        dct_area1=dct_area1[0:8,0:8]#只保留系数矩阵左上角8*8的低频部分
        dct_area1_binary=(dct_area1>np.mean(dct_area1))
        dct_area2=cv2.dct(np.float32(cv2.resize(area_img2,(32,32))))
        dct_area2=dct_area2[0:8,0:8]#只保留系数矩阵左上角8*8的低频部分
        dct_area2_binary=(dct_area2>np.mean(dct_area2))
        hamming_dis=np.sum(dct_area1_binary^dct_area2_binary)

#        cv2.resize(area_img1,(4,4))
#        cv2.imshow('local_img1',area_img1)
#        cv2.imshow('local_img2',area_img2)
#        cv2.waitKey(0)
        
        cv2.circle(area_img1,(int(area_img1.shape[0]/2),int(area_img1.shape[1]/2)),1,255,-1)
        
        cv2.circle(area_img2,(int(area_img2.shape[0]/2),int(area_img2.shape[1]/2)),1,255,-1)
        
        local_img=np.concatenate((area_img1,area_img2),axis=1)
        count+=1
        if count<=56:
            plt.subplot(7,8,count)
            plt.title(hamming_dis)
            plt.imshow(local_img,cmap='gray')#cmap='gray'
            plt.axis('off')
        if i.distance<48 and hamming_dis<9:
            good_matches.append(i)
    
    imgMatch_good=cv2.drawMatches(img_left,keyPoints1,img_right,
                         keyPoints2,good_matches,None,flags=2)
    imgMatch_good_rgb=cv2.cvtColor(imgMatch_good,cv2.COLOR_BGR2RGB)
    plt.figure("imgMatch_good",figsize=(18,6))
    plt.imshow(imgMatch_good_rgb)#cmap='gray'       

#    for i in range(0,h,40):
        #画横线方便观察
#        cv2.line(imgMatch,(0,i),(2*w,i),(0,255,0),1)
#    cv2.namedWindow('imgMatch',0)
#    cv2.imshow('imgMatch',imgMatch)
#    cv2.waitKey(0)

    cv2.destroyAllWindows()