# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:22:23 2019

@author: TOMATO
"""

import cv2
import numpy as np
#import location_bag
import depth




if __name__=="__main__":
    img_left=cv2.imread("d:/python_projects/plastic_bag/img_left1.jpg",-1)
    img_right=cv2.imread("d:/python_projects/plastic_bag/img_right1.jpg",-1)    
    cv2.namedWindow("img_left",0)
    cv2.imshow("img_left",img_left)
    cv2.waitKey(1)
    cv2.namedWindow("img_right",0)
    cv2.imshow("img_right",img_right)
    cv2.waitKey(0)        
    
    #机器人坐标系下点坐标
    P0_rbt=np.array([191.278,371.116,-940.47])
    P1_rbt=np.array([174.326,-253.736,-950.036])
    P2_rbt=np.array([-62.315,22.772,-950.052])
    P3_rbt=np.array([-94.87,-336.171,-952.433])
    P4_rbt=np.array([-263.136,338.908,-947.016])
    P5_rbt=np.array([-281.57,-184.254,-954.64])
    #机器人坐标系下点坐标#不考虑线体不平引起的Z轴误差
#    P0_rbt=np.array([191.278,371.116,-950])
#    P1_rbt=np.array([174.326,-253.736,-950])
#    P2_rbt=np.array([-62.315,22.772,-950])
#    P3_rbt=np.array([-94.87,-336.171,-950])
#    P4_rbt=np.array([-263.136,338.908,-950])
#    P5_rbt=np.array([-281.57,-184.254,-950])    
    
    
    
    
    
    
    
    rbt_arr=np.array([P0_rbt,P1_rbt,P2_rbt,P3_rbt,P4_rbt,P5_rbt]).T
    rbt_arrex=np.concatenate((rbt_arr,np.array([[1,1,1,1,1,1]])),axis=0)
    #计算相机坐标系下各目标的中心点坐标
    center_list=depth.stereo_center3d(img_left,img_right)
    camera_arr=np.array(center_list).T
    camera_arr[[0,1],:]=camera_arr[[1,0],:]#交换x，y轴
    camera_arrex=np.concatenate((camera_arr,np.array([[1,1,1,1,1,1]])),axis=0)
    R_cv2robot=np.dot(rbt_arrex,np.mat(camera_arrex).I)
#    R_cv2robot[3]=[0,0,0,1]
    
    P=(0,0,100,1)
#    
    a=np.dot(R_cv2robot,np.array(P))
    
    b=np.dot(R_cv2robot,np.array([P]).T)
    print(a)
    
    
    cv2.destroyAllWindows()
