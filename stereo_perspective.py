# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:47:03 2019

@author: TOMATO
"""

import numpy as np
import cv2



ROWS=8
COLUMNS=11
SIZE=25

def get_perspectivearr_binocular(img_left,img_right,ROWS=8,COLUMNS=11,SIZE=25):
    #使用棋盘格将双目视觉标定至工作平面
    #输入双目图像，棋盘格内角点行数和列数,每小格长度SIZE(mm)
    #返回投影变换矩阵,dpx,dpy
    h,w,c=img_left.shape
    gray_left=cv2.cvtColor(img_left,cv2.COLOR_BGR2GRAY)
    gray_right=cv2.cvtColor(img_right,cv2.COLOR_BGR2GRAY)
    ret1,corners_left=cv2.findChessboardCorners(gray_left,(ROWS,COLUMNS),None)
    ret2,corners_right=cv2.findChessboardCorners(gray_right,(ROWS,COLUMNS),None)
    if ret1==ret2==True:
        #使用subPix增加点的准确度
        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
        corners_sp_left=cv2.cornerSubPix(gray_left,corners_left,(11,11),(-1,-1),criteria)
        corners_sp_right=cv2.cornerSubPix(gray_right,corners_right,(11,11),(-1,-1),criteria)
        #计算投影变换的输入点，即4个角顶点
        corner_index=[0,7,-8,-1]
        src_persp_left=corners_sp_left[corner_index][:,0,:]
        src_persp_right=corners_sp_right[corner_index][:,0,:]
        #计算中心点
        center_index=int((np.floor(COLUMNS/2))*ROWS+np.floor(ROWS/2))  
        center_left=corners_sp_left[center_index]
        center_right=corners_sp_right[center_index]
        center_y=(center_left[0,1]+center_right[0,1])/2
        #计算中心点所在列上下连线的长度，作为投影变换后的棋盘格高h_chessb
        y_min_index=int((np.floor(COLUMNS/2)+1)*ROWS-1)#
        y_min_point_left=corners_sp_left[y_min_index]
        y_min_point_right=corners_sp_right[y_min_index]
        y_max_index=int((np.floor(COLUMNS/2))*ROWS)#
        y_max_point_left=corners_sp_left[y_max_index]
        y_max_point_right=corners_sp_right[y_max_index]
        h_chessb_left=((y_max_point_left[0,0]-y_min_point_left[0,0])**2+
                       (y_max_point_left[0,1]-y_min_point_left[0,1])**2)**0.5
        h_chessb_right=((y_max_point_right[0,0]-y_min_point_right[0,0])**2+
                       (y_max_point_right[0,1]-y_min_point_right[0,1])**2)**0.5
        h_chessb=(h_chessb_left+h_chessb_right)/2  
        
        #计算中心点所在行左右连线的长度,作为投影变换后的棋盘格宽w_chessb
        x_min_index=int(np.floor(ROWS/2))
        x_min_point_left=corners_sp_left[x_min_index]
        x_min_point_right=corners_sp_right[x_min_index]
        x_max_index=int(-np.floor(ROWS/2))
        x_max_point_left=corners_sp_left[x_max_index]
        x_max_point_right=corners_sp_right[x_max_index]
        w_chessb_left=((x_max_point_left[0,0]-x_min_point_left[0,0])**2+
                       (x_max_point_left[0,1]-x_min_point_left[0,1])**2)**0.5
        w_chessb_right=((x_max_point_right[0,0]-x_min_point_right[0,0])**2+
                       (x_max_point_right[0,1]-x_min_point_right[0,1])**2)**0.5                       
#        w_chessb=max(w_chessb_left,w_chessb_right)  
        w_chessb=(w_chessb_left+w_chessb_right)/2      
        #预标定区域ROI
        dst_persp_left=np.array([[center_left[0,0]-w_chessb/2,center_y+h_chessb*4/7],
                                 [center_left[0,0]-w_chessb/2,center_y-h_chessb*3/7],
                                 [center_left[0,0]+w_chessb/2,center_y+h_chessb*4/7],
                                 [center_left[0,0]+w_chessb/2,center_y-h_chessb*3/7]],np.float32)
        dst_persp_right=np.array([[center_right[0,0]-w_chessb/2,center_y+h_chessb*4/7],
                                  [center_right[0,0]-w_chessb/2,center_y-h_chessb*3/7],
                                  [center_right[0,0]+w_chessb/2,center_y+h_chessb*4/7],
                                  [center_right[0,0]+w_chessb/2,center_y-h_chessb*3/7]],np.float32)
        persp_arr_left=cv2.getPerspectiveTransform(src_persp_left,dst_persp_left)
        persp_arr_right=cv2.getPerspectiveTransform(src_persp_right,dst_persp_right)
        persp_arr=np.array([persp_arr_left,persp_arr_right])
        dpx=SIZE*COLUMNS/w_chessb
        dpy=SIZE*ROWS/h_chessb
    else:
        print("No chessboard found!")
        persp_arr=np.array([])
        dpx,dpy=0,0
    return persp_arr,dpx,dpy


if __name__=="__main__":
    #双目视觉标定至工作平面
    
#    img_left=cv2.imread("d:/python_projects/ZED_imgleft/02left.png",-1)
#    img_right=cv2.imread("d:/python_projects/ZED_imgright/02right.png",-1)
#    img_left=cv2.imread("d:/python_projects/plastic_bag/img_left.jpg",-1)
#    img_right=cv2.imread("d:/python_projects/plastic_bag/img_right.jpg",-1)    
    
    img_left=cv2.imread("d:/python_projects/plastic_bag/chessimg_left.jpg",-1)
    img_right=cv2.imread("d:/python_projects/plastic_bag/chessimg_right.jpg",-1)
    h,w,c=img_left.shape

    gray_left=cv2.cvtColor(img_left,cv2.COLOR_BGR2GRAY)
    gray_right=cv2.cvtColor(img_right,cv2.COLOR_BGR2GRAY)
    
    ret1,corners_left=cv2.findChessboardCorners(gray_left,(ROWS,COLUMNS),None)
    ret2,corners_right=cv2.findChessboardCorners(gray_right,(ROWS,COLUMNS),None)
    if ret1==ret2==True:
        #使用subPix增加点的准确度
        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
        corners_sp_left=cv2.cornerSubPix(gray_left,corners_left,(11,11),(-1,-1),criteria)
        corners_sp_right=cv2.cornerSubPix(gray_right,corners_right,(11,11),(-1,-1),criteria)
        corners_img_left=cv2.drawChessboardCorners(img_left,(ROWS,COLUMNS),corners_sp_left,ret1)
        cv2.namedWindow("corners_img_left",0)
        cv2.imshow("corners_img_left",corners_img_left)
        cv2.waitKey(1)
        corners_img_right=cv2.drawChessboardCorners(img_right,(ROWS,COLUMNS),corners_sp_right,ret2)
        cv2.namedWindow("corners_img_right",0)
        cv2.imshow("corners_img_right",corners_img_right)
        cv2.waitKey(0)
        
        #计算投影变换的输入点，即4个角顶点
        corner_index=[0,7,-8,-1]
        src_persp_left=corners_sp_left[corner_index][:,0,:]
        src_persp_right=corners_sp_right[corner_index][:,0,:]
        for i in range(len(src_persp_left)):
            cv2.circle(corners_img_left,(int(src_persp_left[i][0]),int(src_persp_left[i][1])),8,(0,255,255),-1)
            cv2.circle(corners_img_right,(int(src_persp_right[i][0]),int(src_persp_right[i][1])),8,(0,255,255),-1)
            
        cv2.imshow("corners_img_left",corners_img_left)
        cv2.waitKey(1)
        cv2.imshow("corners_img_right",corners_img_right)           
        cv2.waitKey(0)
        
#        vertices_disparity=src_persp_left-src_persp_right
        #计算中心点
        center_index=int((np.floor(COLUMNS/2))*ROWS+np.floor(ROWS/2))  
        center_left=corners_sp_left[center_index]
        center_right=corners_sp_right[center_index]
        center_left_x=center_left[0,0]
        center_right_x=center_left[0,0]
        center_y=(center_left[0,1]+center_right[0,1])/2
#        disparity_x=center_left_x-center_right_x
        
        cv2.circle(corners_img_left,(int(center_left[0][0]),int(center_left[0][1])),8,(0,0,255),-1)
        cv2.namedWindow("corners_img_left",0)
        cv2.imshow("corners_img_left",corners_img_left)
        cv2.waitKey(1)
        cv2.circle(corners_img_right,(int(center_right[0][0]),int(center_right[0][1])),8,(0,0,255),-1)
        cv2.namedWindow("corners_img_right",0)
        cv2.imshow("corners_img_right",corners_img_right)           
        cv2.waitKey(0)
        
        #计算中心点所在列上下连线的长度，作为投影变换后的棋盘格高h_chessb
        y_min_index=int((np.floor(COLUMNS/2)+1)*ROWS-1)#
        y_min_point_left=corners_sp_left[y_min_index]
        y_min_point_right=corners_sp_right[y_min_index]
        y_max_index=int((np.floor(COLUMNS/2))*ROWS)#
        y_max_point_left=corners_sp_left[y_max_index]
        y_max_point_right=corners_sp_right[y_max_index]
        h_chessb_left=((y_max_point_left[0,0]-y_min_point_left[0,0])**2+
                       (y_max_point_left[0,1]-y_min_point_left[0,1])**2)**0.5
        h_chessb_right=((y_max_point_right[0,0]-y_min_point_right[0,0])**2+
                       (y_max_point_right[0,1]-y_min_point_right[0,1])**2)**0.5
        h_chessb=(h_chessb_left+h_chessb_right)/2
        #显示标定点
        cv2.circle(corners_img_left,(int(y_min_point_left[0][0]),int(y_min_point_left[0][1])),8,(0,255,0),-1)
        cv2.circle(corners_img_left,(int(y_max_point_left[0][0]),int(y_max_point_left[0][1])),8,(0,255,0),-1)
        cv2.namedWindow("corners_img_left",0)
        cv2.imshow("corners_img_left",corners_img_left)
        cv2.waitKey(1)
        cv2.circle(corners_img_right,(int(y_min_point_right[0][0]),int(y_min_point_right[0][1])),8,(0,255,0),-1)
        cv2.circle(corners_img_right,(int(y_max_point_right[0][0]),int(y_max_point_right[0][1])),8,(0,255,0),-1)
        cv2.namedWindow("corners_img_right",0)
        cv2.imshow("corners_img_right",corners_img_right)
        cv2.waitKey(0)
        #计算中心点所在行左右连线的长度,作为投影变换后的棋盘格宽w_chessb
        x_min_index=int(np.floor(ROWS/2))
        x_min_point_left=corners_sp_left[x_min_index]
        x_min_point_right=corners_sp_right[x_min_index]
        x_max_index=int(-np.floor(ROWS/2))
        x_max_point_left=corners_sp_left[x_max_index]
        x_max_point_right=corners_sp_right[x_max_index]
        w_chessb_left=((x_max_point_left[0,0]-x_min_point_left[0,0])**2+
                       (x_max_point_left[0,1]-x_min_point_left[0,1])**2)**0.5
        w_chessb_right=((x_max_point_right[0,0]-x_min_point_right[0,0])**2+
                       (x_max_point_right[0,1]-x_min_point_right[0,1])**2)**0.5                       
#        w_chessb=max(w_chessb_left,w_chessb_right)  
        w_chessb=(w_chessb_left+w_chessb_right)/2         
                       
#        #显示标定点
        cv2.circle(corners_img_left,(int(x_min_point_left[0][0]),int(x_min_point_left[0][1])),8,(255,0,0),-1)
        cv2.circle(corners_img_left,(int(x_max_point_left[0][0]),int(x_max_point_left[0][1])),8,(255,0,0),-1)
        cv2.namedWindow("corners_img_left",0)
        cv2.imshow("corners_img_left",corners_img_left)
        cv2.waitKey(1)
        cv2.circle(corners_img_right,(int(x_min_point_right[0][0]),int(x_min_point_right[0][1])),8,(255,0,0),-1)
        cv2.circle(corners_img_right,(int(x_max_point_right[0][0]),int(x_max_point_right[0][1])),8,(255,0,0),-1)
        cv2.namedWindow("corners_img_right",0)
        cv2.imshow("corners_img_right",corners_img_right)
        cv2.waitKey(0)
        #预标定区域ROI
        dst_persp_left=np.array([[center_left[0,0]-w_chessb/2,center_y+h_chessb*4/7],
                                 [center_left[0,0]-w_chessb/2,center_y-h_chessb*3/7],
                                 [center_left[0,0]+w_chessb/2,center_y+h_chessb*4/7],
                                 [center_left[0,0]+w_chessb/2,center_y-h_chessb*3/7]],np.float32)
        dst_persp_right=np.array([[center_right[0,0]-w_chessb/2,center_y+h_chessb*4/7],
                                  [center_right[0,0]-w_chessb/2,center_y-h_chessb*3/7],
                                  [center_right[0,0]+w_chessb/2,center_y+h_chessb*4/7],
                                  [center_right[0,0]+w_chessb/2,center_y-h_chessb*3/7]],np.float32)
        persp_arr_left=cv2.getPerspectiveTransform(src_persp_left,dst_persp_left)
        persp_arr_right=cv2.getPerspectiveTransform(src_persp_right,dst_persp_right)
        persp_arr=np.array([persp_arr_left,persp_arr_right])
        np.save("persp_arr.npy",persp_arr)
        persp_img_left=cv2.warpPerspective(img_left,persp_arr_left,(w,h))#,borderValue=0
        persp_img_right=cv2.warpPerspective(img_right,persp_arr_right,(w,h))
        cv2.namedWindow("persp_img_left",0)
        cv2.imshow("persp_img_left",persp_img_left)
        cv2.waitKey(1)
        cv2.namedWindow("persp_img_right",0)
        cv2.imshow("persp_img_right",persp_img_right)
        cv2.waitKey(0)
    
        dpx=25*(COLUMNS-1)/w_chessb
        dpy=25*(ROWS-1)/h_chessb


    
    cv2.destroyAllWindows()



