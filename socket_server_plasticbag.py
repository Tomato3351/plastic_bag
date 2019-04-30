# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:50:58 2019

@author: TOMATO
"""

import cv2
import numpy as np
import sys
import socket
import location_bag
sys.path.append("../")
sys.path.append("../MvImport")
import hikvisionCamera
import time



if __name__=="__main__":
    raw_file_path = "AfterConvert_RGB.raw" 
    ptx=836/2448#mm/pix
    pty=709/2048
    device_list = hikvisionCamera.camera_search()
    cam, nPayloadSize = hikvisionCamera.camera_open(device_list, 0)  # 打开相机
    print('open camera successful!')
    hikvisionCamera.load_userset(cam, 1)  # 载入用户设置
    hikvisionCamera.stream_start(cam)#开始取流
    server = socket.socket()
    server.bind(("192.168.125.9",54646))
    server.listen(5)
    cv2.namedWindow("result",0)
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
                elif data==b'start':
                     
                    # 抓取图像
                    start_captureImg = time.perf_counter()  # 计时开始
                    img_data, Frame_num, Width, Height = hikvisionCamera.captureImg_faster(
                        cam, nPayloadSize, raw_file_path, path='', img_name='AfterConvert_BGR.jpg')
                    print('get one frame! Width:', Width, ',Height:', Height)
                    end_captureImg = time.perf_counter()  # 计时结束
                    print('capturing time:', end_captureImg-start_captureImg)
#                    img_data=cv2.imread("d:/python_projects/plastic_bag/imgs/56Left.jpg",1)
                    
                    obj_num,obj_centers,result_img=location_bag.detect_obj(img_data)
                    cv2.destroyWindow("result")
                    cv2.namedWindow("result",0)
                    cv2.imshow("result",result_img)
                    cv2.waitKey(1)
                    if obj_num!=0:
                        #按相机y坐标(机器人x,即线体行进方向)值大小排序,小在前
                        sort_arg_y=np.argsort(obj_centers,axis=0)[:,1]
                        centers_sort_y=[]
                        for i in range(len(obj_centers)):
                            current_index=sort_arg_y[i]
                            centers_sort_y.append((obj_centers[current_index]))

                        send_data=str(obj_num)+','
                        for center in centers_sort_y:
#                            center_x=np.around(center[0]*ptx,3)视觉坐标系下目标x坐标
#                            center_y=np.around(center[1]*pty,3)视觉坐标系下目标y坐标
                            
                            trans_center_x=np.around(354.5-center[1]*ptx,3)#变换至机器人坐标x
                            trans_center_y=np.around(418-center[0]*pty,3)#变换至机器人坐标y

                            send_data+=str(trans_center_x)+','+str(trans_center_y)+',0,'
                        send_data=send_data.rstrip(',')    
                        print(send_data)
                        
                    else:
                        send_data='0'
                        print(send_data)
                    con.send(send_data.encode())
                else:
                    send_data="0"
                    print ("send_data:",send_data)
                    con.send(send_data.encode())
            except ConnectionResetError as e:
                print("err", e)
                break                    
    server.close()                
    hikvisionCamera.stream_stop(cam)#停止取流
    hikvisionCamera.camera_close(cam)#关闭相机   

    cv2.destroyAllWindows()






