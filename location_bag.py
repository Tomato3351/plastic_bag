# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:32:58 2019

@author: TOMATO
"""

import cv2
import glob as gb
import matplotlib.pyplot as plt
import numpy as np

def detect_obj(img):
    #返回待测物体的个数和中心点坐标
    h,w,c=img.shape
    img_Gaussianblur=cv2.GaussianBlur(img,(7,7),0)#参数为高斯矩阵尺寸,标准偏差
    img_r=img_Gaussianblur[:,:,2]#提取R通道 
    gray_level=np.mean(img_r)
    canny_para1=max(5,int(gray_level/3-13))
    canny_para2=max(18,int(gray_level/2+16))
    print(canny_para1,canny_para2)
    edges=cv2.Canny(img_r,canny_para1,canny_para2)
    close_kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))#闭运算核尺寸
    edges_close=cv2.morphologyEx(edges,cv2.MORPH_DILATE,close_kernel,iterations=2)
    contours,hierarchy_L= cv2.findContours(edges_close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#查找轮廓
    centers_obj=[]#储存中心点坐标
    contours_obj=[]
    for cnt in contours:
        M=cv2.moments(cnt)
        area=M['m00']#计算面积
        zero_num=np.where(cnt==0)[0]#轮廓中为0的点
        w_num=np.where(cnt[:,0,0]==w-1)[0]#轮廓中x值为w的点
        h_num=np.where(cnt[:,0,1]==h-1)[0]#轮廓中y值为h的点
        at_edge=len(zero_num) or len(w_num) or len(h_num)
        if area>0.003*h*w and not at_edge:#面积小于阈值的舍弃
            contours_obj.append(cnt)
            cx=M['m10']/M['m00']
            cy=M['m01']/M['m00']
            centers_obj.append((cx,cy))  
    result_img=img.copy()
    cv2.drawContours(result_img,contours_obj,-1, (0,0,255),3)
    for center in centers_obj:
        cv2.circle(result_img,(int(center[0]),int(center[1])),8,(0,0,255),-1)    
#    cv2.imwrite('result.jpg',result_img)
    

    return len(centers_obj),centers_obj,result_img,contours_obj

if __name__=="__main__":
#    img_paths=gb.glob("./imgs\\*.jpg")
#    img_paths=gb.glob("D:/MVS_DATA\\*.jpg")    
#    img=cv2.imread(img_paths[2],1)
#    img=cv2.imread("d:/projects/zed_camera_control/zed_camera_control/capture_image/side/55Left.jpg",1)
    img=cv2.imread("d:/python_projects/plastic_bag/imgs/58Left.jpg",1)
#    img=cv2.imread("AfterConvert_BGR.jpg",1)
    h,w,c=img.shape
    cv2.namedWindow("img",0)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    
    #高斯模糊
    img_Gaussianblur=cv2.GaussianBlur(img,(7,7),0)#参数为高斯矩阵尺寸,标准偏差
    cv2.namedWindow("img_Gaussianblur",0)
    cv2.imshow("img_Gaussianblur",img_Gaussianblur)
    cv2.waitKey(0)    
    
    img_r=img_Gaussianblur[:,:,2]#提取R通道
    gray_level=np.mean(img_r)
    print('mean_gray_level=',gray_level)
    cv2.namedWindow("img_r",0)
    cv2.imshow("img_r",img_r)
    cv2.waitKey(0)
    
    
    canny_para1=max(4,int(gray_level/3-13))
    canny_para2=max(18,int(gray_level/2+16))
    print(canny_para1,canny_para2)
    edges=cv2.Canny(img_r,canny_para1,canny_para2)
    plt.figure("edges",figsize=(15,9))
    plt.imshow(edges,cmap='gray')#cmap='gray'    
    
#    adap_thr = cv2.adaptiveThreshold(img_r,255,0,0,5,-1)        
#    plt.figure("adap_thr",figsize=(15,9))      
#    plt.imshow(adap_thr,cmap='gray')#cmap='gray'        

    close_kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))#闭运算核尺寸
    edges_close=cv2.morphologyEx(edges,cv2.MORPH_DILATE,close_kernel,iterations=2)

    plt.figure("edges_close",figsize=(15,9))      
    plt.imshow(edges_close,cmap='gray')#cmap='gray'
    
    #cv2.connectedComponents
    contours,hierarchy_L= cv2.findContours(edges_close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #RETR_TREE,RETR_EXTERNAL只检测外轮廓
    #hierarchy[i][0],hierarchy[i][1],hierarchy[i][2],hierarchy[i][3]分别指的是Next, Previous, First_child, Parent。
    #hierarchy[i][3]== -1 意为 “轮廓没有父母”， 即“这个轮廓没有上层阶级的轮廓”。
    #hierarchy[i][2]== -1 意为 “此轮廓没有第一个孩子”，即“此轮廓没有下层阶级的轮廓”。    
    drawingboard=np.zeros((h,w,c),np.uint8)
    cv2.drawContours(drawingboard, contours,-1, (0,255, 0),3)   
    plt.figure("drawingboard",figsize=(15,9))      
    plt.imshow(drawingboard,cmap='gray')#cmap='gray'      

##    去除面积小于阈值的轮廓
    contours_obj=[]
    centers_obj=[]
#    area_obj=[]
    for cnt in contours:
        M=cv2.moments(cnt)
        area=M['m00']#计算面积
#        perimeter=cv2.arcLength(cnt,True) #计算周长
        zero_num=np.where(cnt==0)[0]#轮廓中为0的点
        w_num=np.where(cnt[:,0,0]==w-1)[0]#轮廓中x值为w的点
        h_num=np.where(cnt[:,0,1]==h-1)[0]#轮廓中y值为h的点
        at_edge=len(zero_num) or len(w_num) or len(h_num)
        if area>0.005*h*w and not at_edge:#面积小于阈值的舍弃
            contours_obj.append(cnt)
#            area_obj.append(area)
            cx=M['m10']/M['m00']
            cy=M['m01']/M['m00']
            centers_obj.append((cx,cy))
    cv2.drawContours(img,contours_obj,-1, (0,0,255),3)
    for center in centers_obj:
        cv2.circle(img,(int(center[0]),int(center[1])),8,(0,0,255),-1)
    
    rect_list=[]
    for cnt in contours_obj:
        #最小外接矩形
        rect = cv2.minAreaRect(cnt)#返回center,(h,w),angle范围[-90，0)
        rect_list.append(rect)
        cv2.circle(img,(int(rect[0][0]),int(rect[0][1])),8,(255,0,255),-1)#rect中心
        vertices=cv2.boxPoints(rect)
        
        cv2.line(img,(vertices[0][0],vertices[0][1]),(vertices[1][0],vertices[0][1]),(0,255,0), 3)
        text_angle=str(np.around(rect[2]+90,2))
        cv2.putText(img,text_angle,(int(vertices[0][0]-w*0.08),int(vertices[0][1]-h*0.01)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)
        for i in range(4):
            p1=vertices[i,:]
            j=(i+1)%4
            p2=vertices[j,:]
            cv2.line(img,(p1[0],p1[1]),(p2[0],p2[1]),(255,0,255), 3)
            
            text='Point'+str(i)
            
            cv2.putText(img,text,(p1[0],p1[1]),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,0,255),2)
            
            
    print(rect_list)
    cv2.namedWindow("result",0)
    cv2.imshow("result",img)
    cv2.waitKey(0)

    send_data=str(len(contours_obj))+','
    for center in centers_obj:
        center_x=np.around(center[0],3)
        center_y=np.around(center[1],3)

        send_data+=str(center_x)+','+str(center_y)+','
    send_data=send_data.rstrip(',')
    print(send_data)
##     where(area==max(area_list))   
#    maxarea_index = area_list.index(max(area_list))
#    drawingboard=np.zeros((h,w,c),np.uint8)
#    cv2.drawContours(drawingboard, contours[9],-1, (0,255,0),2)  
#    convexhull=cv2.convexHull(contours[maxarea_index])
#    
#    cv2.drawContours(drawingboard, convexhull,-1, (255,255,255),10)       
#    plt.figure("convexhull",figsize=(15,9))      
#    plt.imshow(drawingboard,cmap='gray')#cmap='gray'          
#    
#    approxcurve=cv2.approxPolyDP(contours[maxarea_index],5,True)
#    cv2.drawContours(drawingboard,approxcurve,-1, (255,0,0),10)  
#    plt.figure("approxcurve",figsize=(15,9))      
#    plt.imshow(drawingboard,cmap='gray')#cmap='gray'       
    
    
#    cv2.pointPolygonTest#判断点是否在轮廓内
    
    
    
    cv2.destroyAllWindows()


















