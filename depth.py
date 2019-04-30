# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:48:53 2019

@author: TOMATO
"""

import cv2
import numpy as np
import location_bag
import feature_match


def stereo_depth(img_left,img_right,dpx,dpy):
    #根据视差计算深度,而x和y是根据简单的2d信息计算。
    #ZED双目，单位毫米
#    B=120
#    f=350;dpx=475/1920;dpy=274/1080
    B=120
    f=742
#    dpx=1020/1920;dpy=567/1080
    h,w,c=img_left.shape  

    #检测两图像中存在的目标个数是否一致
    obj_num_left,obj_centers_left,result_img_left,contours_obj_left=location_bag.detect_obj(img_left)
    obj_num_right,obj_centers_right,result_img_right,contours_obj_right=location_bag.detect_obj(img_right)

    result_left=img_left.copy()
    result_right=img_right.copy()
    cv2.drawContours(result_left,contours_obj_left,-1, (0,0,255),3)
#    cv2.circle(result_left,(int(center_left_x),int(center_left_y)),8,(0,0,255),-1)
    cv2.drawContours(result_right,contours_obj_right,-1, (0,0,255),3)
#    cv2.circle(result_right,(int(center_right_x),int(center_right_y)),8,(0,0,255),-1)
    cv2.imwrite("result_left.jpg",result_left)
    cv2.imwrite("result_right.jpg",result_right)    
    
    if obj_num_left!=0 and obj_num_left==obj_num_right:
        #按相机y坐标(机器人x,即线体行进方向)值大小排序,小在前
        sort_arg_left=np.argsort(obj_centers_left,axis=0)[:,1]
        sort_arg_right=np.argsort(obj_centers_right,axis=0)[:,1]
        objects_list=[]
        for i in range(len(obj_centers_left)):
            print('obj:',i)
            #从轮廓得出的形心点坐标，x,y
            current_index_left=sort_arg_left[i]
            current_index_right=sort_arg_right[i]
            center_left_x,center_left_y=obj_centers_left[current_index_left]
            center_right_x,center_right_y=obj_centers_right[current_index_right]
            center_x=(center_left_x+center_right_x)/2
            center_y=(center_left_y+center_right_y)/2
            trans_center_x=np.around(-10-center_y*dpx,3)#变换至机器人坐标x  354.5
            trans_center_y=np.around(590-center_x*dpy,3)#变换至机器人坐标y
            
            #当前目标的轮廓
            contour_left=contours_obj_left[current_index_left]
            contour_right=contours_obj_right[current_index_right]
            #
#            result_left=img_left.copy()
#            result_right=img_right.copy()
#            cv2.drawContours(result_left,contour_left,-1, (0,0,255),3)
#            cv2.circle(result_left,(int(center_left_x),int(center_left_y)),8,(0,0,255),-1)
#            cv2.drawContours(result_right,contour_right,-1, (0,0,255),3)
#            cv2.circle(result_right,(int(center_right_x),int(center_right_y)),8,(0,0,255),-1)
#            cv2.imwrite("result_left.jpg",result_left)
#            cv2.imwrite("result_right.jpg",result_right)
#            cv2.namedWindow("current_left",0)
#            cv2.imshow("current_left",result_left)
#            cv2.waitKey(1)
#            cv2.namedWindow("current_right",0)
#            cv2.imshow("current_right",result_right)
#            cv2.waitKey(1)                    

            #取当前目标的boundingRect
            x1,y1,w1,h1=cv2.boundingRect(contour_left)
            rect_left=img_left[y1:y1+h1,x1:x1+w1,0:3].copy()
            x2,y2,w2,h2=cv2.boundingRect(contour_right)
            rect_right=img_right[y2:y2+h2,x2:x2+w2,0:3].copy()
            
            good_matches,matches,keyPoints1,keyPoints2=feature_match.good_match(rect_left,rect_right)
            imgMatch_good=cv2.drawMatches(rect_left,keyPoints1,rect_right,
                                 keyPoints2,good_matches,None,flags=2)
#            cv2.namedWindow("imgMatch_good",0)
#            cv2.imshow("imgMatch_good",imgMatch_good)
#            cv2.waitKey(1)
            cv2.imwrite("imgMatch.jpg",imgMatch_good)
            incontour_good_matches=[]
            if len(good_matches)!=0:
                points_z=[]
                print('find %d pair of match points' % len(good_matches))
                removecount=0
                for m in good_matches:
                    x1_rect,y1_rect=keyPoints1[m.queryIdx].pt#此点在img_left中的坐标
                    x2_rect,y2_rect=keyPoints2[m.trainIdx].pt#此点在img_right中的坐标
                    point_left=(x1_rect+x1,y1_rect+y1)
                    point_right=(x2_rect+x2,y2_rect+y2)
                    Xleft,Yleft=point_left
                    Xright,Yright=point_right
                    
                    incontour_left=cv2.pointPolygonTest(contour_left,point_left,measureDist=True)#判断点是否在轮廓内
                    incontour_right=cv2.pointPolygonTest(contour_right,point_right,measureDist=True)
                    
                    if incontour_left<38 or incontour_right<38:
                        
                        removecount+=1
                        
#                        good_matches.remove(m)
                    else:
                        ##计算空间坐标，以双目连线中心点为原点
    #                    Yu=h/2-(Yleft+Yright)/2  
                        Zc=B*f/((Xleft-Xright)*dpx)
    #                    Xc=B*(Xleft-w/2)/(Xleft-Xright)-B/2
    #                    Yc=B*Yu*dpy/((Xleft-Xright)*dpx)
    #                    point=(Xc,Yc,Zc)
                        points_z.append(Zc)
                        incontour_good_matches.append(m)
#                        print(Zc)
                print('removecount',removecount)
                #显示轮廓内的点
#                img_incontour_Match_good=cv2.drawMatches(rect_left,keyPoints1,rect_right,
#                                     keyPoints2,incontour_good_matches,None,flags=2)
                
#                cv2.namedWindow("img_incontour_Match_good",0)
#                cv2.imshow("img_incontour_Match_good",img_incontour_Match_good)
#                cv2.waitKey(0)                
                
                pz_len=len(points_z)
                if pz_len>4:
                    #去除z值中的不稳定值(过大或过小)后求平均
                    points_z_mid=points_z[int(1/4*pz_len):int(-1/4*pz_len)]
                    z=np.mean(points_z_mid)
                else:
                    print('len(points_z)==0')
                    z=0.0
            else:
                print('len(good_matches)==0')
                z=0.0
            obj_center=(trans_center_x,trans_center_y,np.around(max(758-z,1),3))
            objects_list.append(obj_center)       
    else:
        print("object number do not match!")
        objects_list=[]
    cv2.destroyAllWindows()
    return objects_list

def stereo_measure3d(img_left,img_right):
    #计算目标3d坐标，先通过中心点视差求坐标，再通过匹配点的平均值求得更加准确的深度
    #ZED双目，单位毫米
#    B=120
#    f=350;dpx=475/1920;dpy=274/1080
    B=120
    f=742;dpx=1002/1920;dpy=566/1080
    h,w,c=img_left.shape
    R_cv2robot=np.array([[9.73886649e-01, -9.67957748e-03,  3.74550333e-02,-9.00866824e+01],
                         [-1.75987481e-02, -9.62287323e-01,  1.61995496e-01,-1.12848425e+02],
                         [2.01693503e-02, -1.95949216e-02, -2.62716023e-01,-7.49965316e+02],
                         [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,1.00000000e+00]])         

    #不考虑线体不平引起的Z轴误差
#    R_cv2robot=np.array([[ 9.73886649e-01, -9.67957748e-03,  3.74550333e-02,-9.00866824e+01],
#                         [-1.75987481e-02, -9.62287323e-01,  1.61995496e-01,-1.12848425e+02],
#                         [ 6.66133815e-16, -6.66133815e-16,  3.55271368e-15,-9.50000000e+02],
#                         [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,1.00000000e+00]])       



      
       
    
    #检测两图像中存在的目标个数是否一致
    obj_num_left,obj_centers_left,result_img_left,contours_obj_left=location_bag.detect_obj(img_left)
    obj_num_right,obj_centers_right,result_img_right,contours_obj_right=location_bag.detect_obj(img_right)

    result_left=img_left.copy()
    result_right=img_right.copy()
    cv2.drawContours(result_left,contours_obj_left,-1, (0,0,255),3)
#    cv2.circle(result_left,(int(center_left_x),int(center_left_y)),8,(0,0,255),-1)
    cv2.drawContours(result_right,contours_obj_right,-1, (0,0,255),3)
#    cv2.circle(result_right,(int(center_right_x),int(center_right_y)),8,(0,0,255),-1)
    cv2.imwrite("result_left.jpg",result_left)
    cv2.imwrite("result_right.jpg",result_right)    
    
    if obj_num_left!=0 and obj_num_left==obj_num_right:
        #按相机y坐标(机器人x,即线体行进方向)值大小排序,小在前
        sort_arg_left=np.argsort(obj_centers_left,axis=0)[:,1]
        sort_arg_right=np.argsort(obj_centers_right,axis=0)[:,1]
        objects_list=[]
        for i in range(len(obj_centers_left)):
            print('obj:',i)
            #从轮廓得出的形心点坐标，x,y
            current_index_left=sort_arg_left[i]
            current_index_right=sort_arg_right[i]
            center_left_x,center_left_y=obj_centers_left[current_index_left]
            center_right_x,center_right_y=obj_centers_right[current_index_right]
            ##计算空间坐标，以双目连线中心点为原点
            centerYu=h/2-(center_left_y+center_right_y)/2  
#            centerZc=B*f/((center_left_x-center_right_x)*dpx)
            centerXc=B*(center_left_x-w/2)/(center_left_x-center_right_x)-B/2
            centerYc=B*centerYu*dpy/((center_left_x-center_right_x)*dpx)
#            center3d=(centerXc,centerYc,centerZc)

            #当前目标的轮廓
            contour_left=contours_obj_left[current_index_left]
            contour_right=contours_obj_right[current_index_right]
            #
#            result_left=img_left.copy()
#            result_right=img_right.copy()
#            cv2.drawContours(result_left,contour_left,-1, (0,0,255),3)
#            cv2.circle(result_left,(int(center_left_x),int(center_left_y)),8,(0,0,255),-1)
#            cv2.drawContours(result_right,contour_right,-1, (0,0,255),3)
#            cv2.circle(result_right,(int(center_right_x),int(center_right_y)),8,(0,0,255),-1)
#            cv2.imwrite("result_left.jpg",result_left)
#            cv2.imwrite("result_right.jpg",result_right)
#            cv2.namedWindow("current_left",0)
#            cv2.imshow("current_left",result_left)
#            cv2.waitKey(1)
#            cv2.namedWindow("current_right",0)
#            cv2.imshow("current_right",result_right)
#            cv2.waitKey(1)                    

#            #计算角度
#            rect_rotate_left = cv2.minAreaRect(contour_left)#返回center,(h,w),angle范围[-90，0)
#            rect_rotate_right = cv2.minAreaRect(contour_right)#返回center,(h,w),angle范围[-90，0)
#            angle=(rect_rotate_left[2]+rect_rotate_right[2])/2

            #取当前目标的boundingRect
            x1,y1,w1,h1=cv2.boundingRect(contour_left)
            rect_left=img_left[y1:y1+h1,x1:x1+w1,0:3].copy()
            x2,y2,w2,h2=cv2.boundingRect(contour_right)
            rect_right=img_right[y2:y2+h2,x2:x2+w2,0:3].copy()
            
            good_matches,matches,keyPoints1,keyPoints2=feature_match.good_match(rect_left,rect_right)
            imgMatch_good=cv2.drawMatches(rect_left,keyPoints1,rect_right,
                                 keyPoints2,good_matches,None,flags=2)
            cv2.namedWindow("imgMatch_good",0)
            cv2.imshow("imgMatch_good",imgMatch_good)
            cv2.waitKey(0)
            cv2.imwrite("imgMatch.jpg",imgMatch_good)
            if len(good_matches)!=0:
                points_z=[]
                print('find %d pair of match points' % len(good_matches))
                removecount=0
                for m in good_matches:
                    x1_rect,y1_rect=keyPoints1[m.queryIdx].pt#此点在img_left中的坐标
                    x2_rect,y2_rect=keyPoints2[m.trainIdx].pt#此点在img_right中的坐标
                    point_left=(x1_rect+x1,y1_rect+y1)
                    point_right=(x2_rect+x2,y2_rect+y2)
                    Xleft,Yleft=point_left
                    Xright,Yright=point_right
                    
                    incontour_left=cv2.pointPolygonTest(contour_left,point_left,measureDist=True)#判断点是否在轮廓内
                    incontour_right=cv2.pointPolygonTest(contour_right,point_right,measureDist=True)
                    
                    if incontour_left<40 or incontour_right<40:
                        removecount+=1
                    else:
                        ##计算空间坐标，以双目连线中心点为原点
                        Yu=h/2-(Yleft+Yright)/2  
                        Zc=B*f/((Xleft-Xright)*dpx)
                        Xc=B*(Xleft-w/2)/(Xleft-Xright)-B/2
                        Yc=B*Yu*dpy/((Xleft-Xright)*dpx)
                        point_current=(Yc,Xc,Zc,1)#x,y轴交换过，为了转换为右手坐标系
                        
                        point_current_robot=np.dot(R_cv2robot,np.array(point_current))
    #                    point=(Xc,Yc,Zc)
                        points_z.append(point_current_robot[2])
                        print('Zc=',Zc,'z=',point_current_robot[2])
                print('removecount',removecount)
                if len(points_z)!=0:
                    z=np.around(np.mean(points_z),3)
                    
                else:
                    print('len(points_z)==0')
                    z=0.0
            else:
                print('len(good_matches)==0')
                z=0.0
                

            trans_center_x=np.around(354.5-centerYc*dpx,3)#变换至机器人坐标x
            trans_center_y=np.around(418-centerXc*dpy,3)#变换至机器人坐标y
            obj_center=(trans_center_x,trans_center_y,z)#
            objects_list.append(obj_center)       
    else:
        print("object number do not match!")
        objects_list=[]
    cv2.destroyAllWindows()
    return objects_list

def stereo_center3d(img_left,img_right):
    #计算各目标中心点的3d信息，无特征匹配
    #ZED双目，单位毫米
#    B=120
#    f=350;dpx=475/1920;dpy=274/1080
    B=120
    f=742;dpx=1020/1920;dpy=567/1080
    h,w,c=img_left.shape
    #检测两图像中存在的目标个数是否一致
    obj_num_left,obj_centers_left,result_img_left,contours_obj_left=location_bag.detect_obj(img_left)
    obj_num_right,obj_centers_right,result_img_right,contours_obj_right=location_bag.detect_obj(img_right)

    result_left=img_left.copy()
    result_right=img_right.copy()
    cv2.drawContours(result_left,contours_obj_left,-1, (0,0,255),3)
#    cv2.circle(result_left,(int(center_left_x),int(center_left_y)),8,(0,0,255),-1)
    cv2.drawContours(result_right,contours_obj_right,-1, (0,0,255),3)
#    cv2.circle(result_right,(int(center_right_x),int(center_right_y)),8,(0,0,255),-1) 
    objects_list=[]
    if obj_num_left!=0 and obj_num_left==obj_num_right:
        #按相机y坐标(机器人x,即线体行进方向)值大小排序,小在前
        sort_arg_left=np.argsort(obj_centers_left,axis=0)[:,1]
        sort_arg_right=np.argsort(obj_centers_right,axis=0)[:,1]
        for i in range(len(obj_centers_left)):
            print('obj:',i)
            #从轮廓得出的形心点坐标，x,y
            current_index_left=sort_arg_left[i]
            current_index_right=sort_arg_right[i]
            Xleft,Yleft=obj_centers_left[current_index_left]
            Xright,Yright=obj_centers_right[current_index_right]
            #计算空间坐标，以双目连线中心点为原点
            Yu=h/2-(Yleft+Yright)/2  
            Zc=B*f/((Xleft-Xright)*dpx)
            Xc=B*(Xleft-w/2)/(Xleft-Xright)-B/2
            Yc=B*Yu*dpy/((Xleft-Xright)*dpx)
            point=(Xc,Yc,Zc)
            print('point',point)
            objects_list.append(point)
    else:
        print("object number do not match!")
    cv2.destroyAllWindows()
    return objects_list    
    

if __name__=="__main__":
    #ZED双目，单位毫米
#    B=120
#    f=350;dpx=475/1920;dpy=274/1080
    B=120
    f=742;dpx=1020/1920;dpy=567/1080
#    img_left=cv2.imread("d:/python_projects/plastic_bag/imgs/56Left.jpg",-1)
#    img_right=cv2.imread("d:/python_projects/plastic_bag/imgs/56Right.jpg",-1)
    img_left=cv2.imread("d:/python_projects/plastic_bag/img_left.jpg",-1)
    img_right=cv2.imread("d:/python_projects/plastic_bag/img_right.jpg",-1)    
    
    cv2.namedWindow("img_left",0)
    cv2.imshow("img_left",img_left)
    cv2.waitKey(1)
    cv2.namedWindow("img_right",0)
    cv2.imshow("img_right",img_right)
    cv2.waitKey(0)
    h,w,c=img_left.shape
    #检测到的目标数目，目标中心点坐标，结果图，轮廓
    obj_num_left,obj_centers_left,result_img_left,contours_obj_left=location_bag.detect_obj(img_left)
    cv2.namedWindow("result_img_left",0)
    cv2.imshow("result_img_left",result_img_left)
    cv2.waitKey(1)
    obj_num_right,obj_centers_right,result_img_right,contours_obj_right=location_bag.detect_obj(img_right)
    
    cv2.namedWindow("result_img_right",0)
    cv2.imshow("result_img_right",result_img_right)
    cv2.waitKey(0)    

    #按相机y坐标(机器人x,即线体行进方向)值大小排序,小在前
    sort_arg_y=np.argsort(obj_centers_left,axis=0)[:,1]
    centers_left_sort=[]
    centers_right_sort=[]
    contours_left_sort=[]
    contours_right_sort=[]
    points=[]
    point_string=''
    pointsleft_incontour_list=[]
    pointsright_incontour_list=[]
    for i in range(len(obj_centers_left)):
        current_index=sort_arg_y[i]
        #当前目标的中心点
        center_left=obj_centers_left[current_index]
        centers_left_sort.append(center_left)
        center_right=obj_centers_right[current_index]
        centers_right_sort.append(center_right)
        
        #当前目标的轮廓
        contour_left=contours_obj_left[current_index]
        contours_left_sort.append(contour_left)
        contour_right=contours_obj_right[current_index]
        contours_right_sort.append(contour_right)
        #显示当前目标的轮廓和形心
        result_left=img_left.copy()
        result_right=img_right.copy()
        cv2.drawContours(result_left,contour_left,-1, (0,0,255),3)
        cv2.circle(result_left,(int(center_left[0]),int(center_left[1])),8,(0,0,255),-1)
        cv2.drawContours(result_right,contour_right,-1, (0,0,255),3)
        cv2.circle(result_right,(int(center_right[0]),int(center_right[1])),8,(0,0,255),-1)
        cv2.namedWindow("current_left",0)
        cv2.imshow("current_left",result_left)
        cv2.waitKey(1)
        cv2.namedWindow("current_right",0)
        cv2.imshow("current_right",result_right)
        cv2.waitKey(1)        
        #取当前目标的boundingRect
        x1,y1,w1,h1=cv2.boundingRect(contour_left)
        rect_left=img_left[y1:y1+h1,x1:x1+w1]
        x2,y2,w2,h2=cv2.boundingRect(contour_right)
        rect_right=img_right[y2:y2+h2,x2:x2+w2]
        cv2.namedWindow("rect_left",0)
        cv2.imshow("rect_left",rect_left)
        cv2.waitKey(1)
        cv2.namedWindow("rect_right",0)
        cv2.imshow("rect_right",rect_right)
        cv2.waitKey(0)

        good_matches,matches,keyPoints1,keyPoints2=feature_match.good_match(rect_left,rect_right)
        
        imgMatch_good=cv2.drawMatches(rect_left,keyPoints1,rect_right,
                             keyPoints2,good_matches,None,flags=2)
        cv2.namedWindow("imgMatch_good",0)
        cv2.imshow("imgMatch_good",imgMatch_good)
        cv2.waitKey(1)
        
        incontour_good_matches=[]
        if len(good_matches)!=0:
            print('find %d pair of match points' % len(good_matches))
            points_current=[]
            pointsleft_incontour=[]
            pointsright_incontour=[]
            removecount=0
            points_z=[]
            for m in good_matches:
                x1_rect,y1_rect=keyPoints1[m.queryIdx].pt#此点在img_left中的坐标
                x2_rect,y2_rect=keyPoints2[m.trainIdx].pt#此点在img_right中的坐标
                point_left=(x1_rect+x1,y1_rect+y1)
                point_right=(x2_rect+x2,y2_rect+y2)
                Xleft,Yleft=point_left
                Xright,Yright=point_right
                
                incontour_left=cv2.pointPolygonTest(contour_left,point_left,measureDist=True)#判断点是否在轮廓内
                pointsleft_incontour.append(incontour_left)
                incontour_right=cv2.pointPolygonTest(contour_right,point_right,measureDist=True)
                pointsright_incontour.append(incontour_right)
                
                if incontour_left<38 or incontour_right<38:
#                    good_matches.remove(m)
                    removecount+=1
                    
                else:
                    incontour_good_matches.append(m)
                    #计算空间坐标，以双目连线中心点为原点
                    Yu=h/2-(Yleft+Yright)/2  
                    Zc=B*f/((Xleft-Xright)*dpx)
                    Xc=B*(Xleft-w/2)/(Xleft-Xright)-B/2
                    Yc=B*Yu*dpy/((Xleft-Xright)*dpx)
                    point=(Xc,Yc,Zc)
                    points_z.append(Zc)
                    points_current.append(point)
                    point_string+='\n'+str(Xc)+' '+str(Yc)+' '+str(Zc)
                    print('Zc=',Zc)
            print('remove',removecount,'edge points')
            print('z=',np.mean(points_z))

            points.append(points_current)
            pointsleft_incontour_list.append(pointsleft_incontour)   
            pointsright_incontour_list.append(pointsright_incontour)   
            
        imgMatch_incontourgood=cv2.drawMatches(rect_left,keyPoints1,rect_right,
                             keyPoints2,incontour_good_matches,None,flags=2)
        cv2.namedWindow("imgMatch_incontourgood",0)
        cv2.imshow("imgMatch_incontourgood",imgMatch_incontourgood)
        cv2.waitKey(0)            
            
            
    
    header="# .PCD v0.7 - Point Cloud Data file format\n\
    VERSION 0.7\n\
    FIELDS x y z\n\
    SIZE 4 4 4\n\
    TYPE F F F\n\
    COUNT 1 1 1\n\
    WIDTH width\n\
    HEIGHT 1\n\
    VIEWPOINT 0 0 0 1 0 0 0\n\
    POINTS points\n\
    DATA ascii"        
    point_num=sum(len(points[i]) for i in range(len(points)))
    header=header.replace("width",str(point_num))
    header=header.replace("points",str(point_num))        

#    f_pcd=open('point_cloud.pcd','w+')
#    f_pcd.write(header)
#    f_pcd.write(point_string)
#    f_pcd.close()
    
    
    







    cv2.destroyAllWindows()