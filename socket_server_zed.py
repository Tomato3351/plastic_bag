# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:00:40 2019

@author: TOMATO
"""

import cv2
import numpy as np
import pyzed.sl as sl
import socket
#import location_bag
import depth
import stereo_perspective
import time
import sys
#import msvcrt

#
#        # Grab an image, a RuntimeParameters object must be given to grab()
#        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
#            # A new image is available if grab() returns SUCCESS
#            zed.retrieve_image(image_left, sl.VIEW.VIEW_LEFT)
#            
#            zed.retrieve_image(image_right, sl.VIEW.VIEW_RIGHT)
#            img_leftdata=image_left.get_data()
#            img_rightdata=image_right.get_data()
#            cv2.imshow("left_live",img_leftdata)
#            cv2.imshow("right_live",img_rightdata)
#            c=cv2.waitKey(1)&0xFF
#            if c==27 or c==113:
#                break
if __name__=="__main__":

#    # Create a Camera object
    zed = sl.Camera()
##    
##    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
##    init_params.sdk_verbose = False
##    # Use HD1080 video mode
    init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080    #HD1080  
    zed.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE,100,False)#,BRIGHTNESS
    init_params.camera_fps = 60  # Set fps at 60
##
    init_params.sdk_gpu_id=0#default'-1',best device found
##    # Open the camera
    try:
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            sys.exit(0)
    except:
        print('err:',err)
    if err==sl.ERROR_CODE.SUCCESS:
        zed.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE,50,False)#,BRIGHTNESS
        image_left = sl.Mat()
        image_right = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()#抓取图像所需实时参数
##    cv2.namedWindow("left_live",0)
##    cv2.namedWindow("right_live",0)
##    
        #载入投影标定信息
        persp_arr,dpx,dpy=np.load("persp_arr20190423.npy")
        persp_arr_left,persp_arr_right=persp_arr        
        
##    #创建socket对象

        server = socket.socket()
#        server.bind(("127.0.0.1",54646))
        server.bind(("192.168.125.9",54646))
        server.listen(5)
        while True:
            print("Waiting for connection...")
            order=input("Any key to continue,q to exit!\n>").strip()##去除字符串首尾空格
            if order=='q' or order=='exit':
                break
            else:
                print('Continue...')
            con,addr = server.accept()
            print("Connected by:",addr)
            while True:
                try:
                #con 就是客户端连过来，在服务器生成的一个连接实例。
                    data = con.recv(1024)
                    if not data:
                        break
                    print('receive:',data)
                    if data==b'exit':
                        time.sleep(2)
                        break
                    elif data==b'cal':#获取投影变换矩阵，标定
                        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
#                            # A new image is available if grab() returns SUCCESS
                            start_captureImg = time.perf_counter()  # 抓图计时开始
                            zed.retrieve_image(image_left, sl.VIEW.VIEW_LEFT)
                            zed.retrieve_image(image_right, sl.VIEW.VIEW_RIGHT)
                            img_leftdata=image_left.get_data()[:,:,:3]
                            img_rightdata=image_right.get_data()[:,:,:3]
                            cv2.imwrite('chessimg_left.jpg',img_leftdata)
                            cv2.imwrite('chessimg_right.jpg',img_rightdata)
                            end_captureImg = time.perf_counter()  # 抓图计时结束
                            print('capturing time:', end_captureImg-start_captureImg)
                            persp_arr,dpx,dpy=stereo_perspective.get_perspectivearr_binocular(
                                    img_leftdata,img_rightdata,ROWS=8,COLUMNS=11)
                            np.save("persp_arr.npy",[persp_arr,dpx,dpy])
                            if len(persp_arr)!=0:
                                send_data='calibrate successful,please restart the server'
                            else:
                                send_data='calibrate failed'
                            print ("send_data:",send_data)
                            con.send(send_data.encode())
                    elif data==b'start': 
#                            # Grab an image, a RuntimeParameters object must be given to grab()
                        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
#                            # A new image is available if grab() returns SUCCESS
                            start_captureImg = time.perf_counter()  # 抓图计时开始
                            zed.retrieve_image(image_left, sl.VIEW.VIEW_LEFT)
                            zed.retrieve_image(image_right, sl.VIEW.VIEW_RIGHT)
                            img_leftdata=image_left.get_data()[:,:,:3]
                            img_rightdata=image_right.get_data()[:,:,:3]  
                            h,w,c=img_leftdata.shape
                            cv2.imwrite('img_left.jpg',img_leftdata)
                            cv2.imwrite('img_right.jpg',img_rightdata)
                            end_captureImg = time.perf_counter()  # 抓图计时结束
                            print('capturing time:', end_captureImg-start_captureImg)
                            start_algorithm = time.perf_counter()  # 算法计时开始
                            img_left=cv2.warpPerspective(img_leftdata,persp_arr_left,(w,h))#,borderValue=0
                            img_right=cv2.warpPerspective(img_rightdata,persp_arr_right,(w,h))
                            cv2.imwrite('img_left_persp.jpg',img_left)
                            cv2.imwrite('img_right_persp.jpg',img_right)
                            center_list=depth.stereo_depth(img_left,img_right,dpx,dpy)
                            end_algorithm = time.perf_counter()  # 算法计时结束
                            print('algorithm time:', end_algorithm-start_algorithm)
                            if len(center_list)!=0:
                                send_data=str(len(center_list))
                                for center in center_list:
                                    send_data+=','+str(center[0])+','+str(center[1])+','+str(center[2])
        #                           send_data=send_data.rstrip(',')
                            else:
                                send_data="0"         
                        else:
                            print('grab image error!')
                            send_data="0"
                        print ("send_data:",send_data)
                        con.send(send_data.encode())
                    else:
                        send_data="0"
                        print ("send_data:",send_data)
                        con.send(send_data.encode())
                except ConnectionResetError as e:
                    print("err", e)
                    break
        server.close()
        err = zed.close()# Close the camera
        cv2.destroyAllWindows()
        
        
#    img_left=cv2.imread("d:/python_projects/plastic_bag/imgs/59Left.jpg",-1)
#    img_right=cv2.imread("d:/python_projects/plastic_bag/imgs/59Right.jpg",-1)
#    center_list=depth.stereo_measure(img_left,img_right)

#    print(center_list)
#    
#    send_data=str(len(center_list))
#
#    for center in center_list:
#        send_data+=','+str(center[0])+','+str(center[1])+','+str(center[2])
#        
#    
##    send_data=send_data.rstrip(',')
#    print(send_data)
    
#    ch=msvcrt.getch()
    
    