from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import argparse
import sys
from collections import deque
import torch
from glob import glob
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

import threading
from threading import Thread

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
tracker_config="../pysot/experiments/siamrpn_mobilev2_l234_dwxcorr/config.yaml"
snapshot="../pysot/experiments/siamrpn_mobilev2_l234_dwxcorr/model.pth"

netMain = None
metaMain = None
altNames = None

center_box_thresh = 0.10
iou_thresh = 0.3
delta_thresh = 1.02
dist_thresh = 50
alarm_thresh = 2
buffer_size = 5 

debug = False
print_time = False

NUM_BUFFER = 3
sem = []
sem.append(threading.Semaphore(1))
sem.append(threading.Semaphore(1))
sem.append(threading.Semaphore(1))

vidList = glob('../vidList/*')
vidList.sort()

frame_buffer = []
total_frame = 0
height = 0
width = 0

line_time = 0
yolo_time = 0
tracking_time = 0

center_box_0 = None
center_pt1_0 = None
center_pt2_0 = None

center_box_1 = None
center_pt1_1 = None
center_pt2_1 = None

center_box_2 = None
center_pt1_2 = None
center_pt2_2 = None

prev_bbox_0 = [0,0,0,0]
prev_bbox_1 = [0,0,0,0]
prev_bbox_2 = [0,0,0,0]

yolo_fail_0 = True
yolo_fail_1 = True
yolo_fail_2 = True


class carObject:
    def __init__(self):
        self.bbox_history = deque(maxlen=buffer_size)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices, color3=(255,255,255), color1=255):
    mask = np.zeros_like(img) 
    if len(img.shape) > 2:
        color = color3
    else: 
        color = color1

    cv2.fillPoly(mask, vertices, color)
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    return lines


def weighted_img(img, initial_img, a=1, b=1., r=0.):
    return cv2.addWeighted(initial_img, a, img, b, r)


def get_fitline(img, f_lines):
    lines = np.squeeze(f_lines)
    lines = lines.reshape(lines.shape[0]*2,2)
    output = cv2.fitLine(lines,cv2.DIST_L2,0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0]-1)-y)/vy*vx + x) , img.shape[0]-1
    x2, y2 = int(((img.shape[0]/2+100)-y)/vy*vx + x) , int(img.shape[0]/2+100)
    
    result = [x1,y1,x2,y2]
    return result


def getCrossPt(line1,line2):
    m1=getSlope(line1)
    m2=getSlope(line2)
    if m1==m2:
        #print('parallel')
        return None
    cx = (line1[0] * m1 - line1[1] - line2[0] * m2 + line2[1]) / (m1 - m2)
    cy = m1 * (cx - line1[0]) + line1[1]
    return int(cx), int(cy)


def makeLines(image,height,width):
    #gray_img=grayscale(image)
    gray_img=image
    blur_img = gaussian_blur(gray_img, 3)
    canny_img = canny(blur_img, 50, 150)
    #vertices = np.array([[(50,height),(width/2-45, height/3), (width/2+45, height/3), (width-50,height),(width/2,height*3/4)]], dtype=np.int32)
    vertices = np.array([[(50,height),(width/2-45, height/3), (width/2+45, height/3), (width-50,height)]], dtype=np.int32)
    ROI_img = region_of_interest(canny_img, vertices)
    line_arr = hough_lines(ROI_img, 1, 1 * np.pi/180, 30, 10, 20)
    #cv2.imshow('roi',ROI_img)
    
    return line_arr


def limitSlope(line_arr):
    line_arr = np.squeeze(line_arr)
    if(line_arr.ndim== 1):
        slope_degree=(np.arctan2(line_arr[1]-line_arr[3],line_arr[0]-line_arr[2])*180)/np.pi
    else : slope_degree = (np.arctan2(line_arr[:,1] - line_arr[:,3], line_arr[:,0] - line_arr[:,2]) * 180) / np.pi
    line_arr = line_arr[np.abs(slope_degree)<155]
    slope_degree = slope_degree[np.abs(slope_degree)<155]
    line_arr = line_arr[np.abs(slope_degree)>105]
    slope_degree = slope_degree[np.abs(slope_degree)>105]
    L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
    L_lines, R_lines = L_lines[:,None], R_lines[:,None]
    return L_lines,R_lines


def checkSameLine(line_list,newPt):
    if len(line_list)==0:
        return True
    x1=newPt[0][0]
    y1=newPt[0][1]
    x2=newPt[0][2]
    y2=newPt[0][3]
    m1=(y2-y1)/(x2-x1)
    if (m1<0): m1=-m1
    
    x3=line_list[0][0][0]
    y3=line_list[0][0][1]
    x4=line_list[0][0][2]
    y4=line_list[0][0][3]
    m2=(y4-y3)/(x4-x3)
    if (m2<0): m2=-m2
    
    if m1-m2 > 0.3 or m1-m2 <-0.3 :
        #print("It's different")
        return False

    return True


def getSlope(line):
    x11=line[0]
    y11=line[1]
    x12=line[2]
    y12=line[3]
    m1 = (y12 - y11) / (x12 - x11)
    return m1


def getSlopeAndIntercept(line):
    m1=getSlope(line)
    b = -line[0]*m1+line[1]
    return m1,b


def getFitlinePt(image,lines):
    del lines[0]
    lines=np.array(lines)
    fit_line = get_fitline(image,lines)
    lines=lines.tolist()
    flag=True
    return flag,fit_line


def checkInsideLanes(line1,line2,pt_x,pt_y):
    m1,b1=getSlopeAndIntercept(line1)
    m2,b2=getSlopeAndIntercept(line2)
    
    result= True if (m1*pt_x+b1 - pt_y)*(m2*pt_x+b2 - pt_y)>0 else False
    return result


def checkDist(lines,new_line):
    #lines= normalized vector vx,vy , point x,y
    if len(lines) == 0 : return True
    #print('\t \t line is',lines)
    vectorU=np.array([lines[0]-lines[2],lines[1]-lines[3]])
    vectorV=np.array([lines[0]-new_line[0][0][0],lines[1]-new_line[0][0][1]])
    
    numerator= abs(np.cross(vectorU,vectorV))
    denominator=np.linalg.norm(vectorU)
    #print(numerator,"nu is",denominator,"deno is")
    
    distance= numerator/denominator
    #print('\t dist is',distance)
    if distance < dist_thresh : return True
    return False


def LineDetector(image,height,width,l_lines,r_lines,left_fit,right_fit):
    line_arr=makeLines(image,height,width)

    if line_arr is None:
        return l_lines,r_lines,left_fit,right_fit

    L_lines,R_lines=limitSlope(line_arr)
    """
    for line in L_lines:
        x1=line[0][0]
        y1=line[0][1]
        x2=line[0][2]
        y2=line[0][3]
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    
    for line in R_lines:
        x1=line[0][0]
        y1=line[0][1]
        x2=line[0][2]
        y2=line[0][3]
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    """
    if len(L_lines)>0:
        if L_lines[0][0][0]<(width/2) and L_lines[0][0][2]<(width/2) and checkDist(left_fit,L_lines):
            #cv2.circle(image,(L_lines[0][0][0],L_lines[0][0][1]),15,(0,255,0),15)
            l_lines.append(L_lines[0])
            #print('left add~~~')
    if len(R_lines)>0:
        if R_lines[0][0][0]>(width/2) and R_lines[0][0][2]>(width/2) and checkDist(right_fit,R_lines):
            r_lines.append(R_lines[0])
            #cv2.circle(image,(R_lines[0][0][0],R_lines[0][0][1]),10,(255,255,0),10)
            #print('right add~~')
    
    return l_lines,r_lines,left_fit,right_fit


def PYSOTINIT():
    # load config
    cfg.merge_from_file(tracker_config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)
    return tracker


def convertBack(x, y, w, h):
    xmin = float(x - (w / 2))
    xmax = float(x + (w / 2))
    ymin = float(y - (h / 2))
    ymax = float(y + (h / 2))
    return xmin, ymin, xmax, ymax


def alarm(prev_bbox, cur_bbox, alarmCnt):
    if prev_bbox != [0,0,0,0]:
        cur_delta_bbox = get_area(cur_bbox) / get_area(prev_bbox)
        if cur_delta_bbox > delta_thresh:
            alarmCnt += 1
        else:
            if alarmCnt > 0:
                alarmCnt -= 1
            else:
                alarmCnt = 0
    
    return alarmCnt


def get_area(bbox):
    return (bbox[3]-bbox[1])*(bbox[2]-bbox[0])


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = (xB - xA + 1) * (yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    if xA >= xB or yA >= yB:
        iou = 0
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea + 0.1)
   
    return iou


def cvDrawBoxes(detections, img, width, height, center_index):
    if center_index != None:
        x, y, w, h = detections[center_index][2][0] * width / darknet.network_width(netMain),\
            detections[center_index][2][1] * height / darknet.network_height(netMain),\
            detections[center_index][2][2] * width / darknet.network_width(netMain),\
            detections[center_index][2][3] * height / darknet.network_height(netMain)
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))

            #if img.shape[1]*0.45 < x < img.shape[1]*0.55:
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 3)
    return img


def jbDrawBoxes(detections, img, width, height, center_index):
    if center_index != None:
        xmin, ymin, xmax, ymax = detections[center_index][0],\
            detections[center_index][1] ,\
            detections[center_index][2] ,\
            detections[center_index][3] 

        pt1 = (int(round(xmin)), int(round(ymin)))
        pt2 = (int(round(xmax)), int(round(ymax)))
        cv2.rectangle(img, pt1, pt2, (255, 0, 0), 3)
    return img


def YOLO():
    global metaMain, netMain, altNames
    configPath = "/home/dohe0342/darknet/cfg/yolov3.cfg"
    weightPath = "/home/dohe0342/darknet/bin/yolov3.weights"
    metaPath = "/home/dohe0342/darknet/cfg/coco.data"

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass


def center_box_decider(frame_read, l_lines, r_lines, left_fit_line, right_fit_line, not_real_bbox, center_x, center_y):
    left_flag = False
    right_flag = False
    
    if len(l_lines)==10:
        left_flag,left_fit_line=getFitlinePt(frame_read,l_lines)
    if len(r_lines)==10:
        right_flag,right_fit_line=getFitlinePt(frame_read,r_lines)

    if left_flag == True and right_flag == True:
        center_x,center_y=getCrossPt(left_fit_line,right_fit_line)
    
    """
    cv2.circle(frame_read,(center_x,center_y),10,(0,0,255),5,8)
    if len(left_fit_line)!=0:
        draw_fit_line(frame_read,left_fit_line)
    if len(right_fit_line)!=0:
        draw_fit_line(frame_read,right_fit_line)
    """

    center_box_vir = (width*0.48, height*0.3, width*0.53, height*0.4)
    center_pt1_vir = (int(center_box_vir[0]),int(center_box_vir[1]))
    center_pt2_vir = (int(center_box_vir[2]),int(center_box_vir[3]))

    if center_x != None and center_y != None:
        center_box_real = (center_x-width*0.03, center_y-height*0.03,center_x+width*0.03,center_y+height*0.10)
        center_pt1_real = (int(center_box_real[0]),int(center_box_real[1]))
        center_pt2_real = (int(center_box_real[2]),int(center_box_real[3]))
        compare_center = bb_intersection_over_union(center_box_vir, center_box_real)
        #print(compare_center)
        if compare_center < center_box_thresh:
            not_real_bbox = True
        else:
            not_real_bbox = False
        if not_real_bbox == True:
            if debug:
                print(str(frame_number) + "out of center set virtual")
            center_box = center_box_vir
            center_pt1 = (int(center_box[0]),int(center_box[1]))
            center_pt2 = (int(center_box[2]),int(center_box[3]))

        else:
            if debug:
                print(str(frame_number) + "set real")
            center_box = center_box_real
            center_pt1 = (int(center_box[0]),int(center_box[1]))
            center_pt2 = (int(center_box[2]),int(center_box[3]))

    else:
        if debug:
            print(str(frame_number) + "no lane detection set virtual")
        center_box = center_box_vir
        center_pt1 = (int(center_box[0]),int(center_box[1]))
        center_pt2 = (int(center_box[2]),int(center_box[3]))

        #print(not_real_bbox)

    return center_box, center_pt1, center_pt2, not_real_bbox, l_lines, r_lines, left_fit_line, right_fit_line, center_x, center_y


def draw_and_show(frame_read, frame_number, center_pt1, center_pt2, cur_bbox, alarmCnt):
    cv2.putText(frame_read, str(frame_number),
                (width - 70, 40), cv2.FONT_HERSHEY_SIMPLEX,
                 1, (0, 0, 255), 3)
    cv2.rectangle(frame_read, center_pt1, center_pt2, (0, 0, 255), 3)
    
    if cur_bbox !=[0,0,0,0]:
        cv2.rectangle(frame_read, (int(round(cur_bbox[0])), 
            int(round(cur_bbox[1]))), 
            (int(round(cur_bbox[2])), int(round(cur_bbox[3]))),
            (0, 255, 0), 3)
        if alarmCnt > alarm_thresh:
            cv2.putText(frame_read, "alarm alarm!!",
                       (70, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)


    """
    try:
        cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
        cv2.imshow('Demo', frame_read)
        ch=cv2.waitKey(1)
        if ch==32:
            while 1:
                ch = cv2.waitKey(3)
                if ch == 32: break
    except:
        print("Display Error")
        sys.exit()
    """

    return frame_read


def yolo_detect(frame_read, center_box, yolo_fail, car_buffer):
    iou = []
    bbox_list = list()
    frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb,
                               (darknet.network_width(netMain),
                                darknet.network_height(netMain)),
                               interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
    detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.30)
    
    for detection in detections:
        if detection[0] == b'car' or detection[0] == b'truck' or detection[0] == b'bus':
            x, y, w, h = detection[2][0] * width / darknet.network_width(netMain),\
                detection[2][1] * height / darknet.network_height(netMain),\
                detection[2][2] * width / darknet.network_width(netMain),\
                detection[2][3] * height / darknet.network_height(netMain)
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            
            yolo_bbox = [xmin, ymin, xmax, ymax]
            bbox_iou=bb_intersection_over_union(center_box, yolo_bbox)
            if bbox_iou > 0:
                iou.append(bbox_iou)
                bbox_list.append(yolo_bbox)

    if len(bbox_list) != 0:
        #cur_bbox=bbox_list[iou.index(max(iou))]
        car_buffer.bbox_history.append(bbox_list[iou.index(max(iou))])
        yolo_fail = False
        return bbox_list[iou.index(max(iou))], yolo_fail
    else:
        yolo_fail = True
        return [0,0,0,0], yolo_fail

class MyThread(Thread):
    def __init__(self, tid, darknet_image, tracker):
        Thread.__init__(self)
        self.tid = tid
        self.input_idx = 0
        self.buf_idx = 0
        self.b_ready_second_process = False
        self.b_ready_third_process = False

        self.darknet_image = darknet_image
        self.tracker = tracker

        self.l_lines = []
        self.r_lines = []
        self.left_fit_line = []
        self.right_fit_line = []
        self.center_x = None
        self.center_y = None

        self.car_buffer = carObject()

        self.prev_bbox = [0,0,0,0]
        self.cur_bbox = [0,0,0,0]
        self.cur_delta_bbox = 0
        self.alarmCnt = 0

    def run(self):
        global center_box_0, center_box_1, center_box_2, center_pt1_0, center_pt2_0, center_pt1_1, center_pt2_1, center_pt1_2, center_pt2_2, yolo_fail_0, yolo_fail_1, yolo_fail_2, prev_bbox_0, prev_bbox_1, prev_bbox_2, line_time, yolo_time, tracking_time
        while self.input_idx < total_frame:
            if self.tid == 0:
                sem[self.buf_idx].acquire()
                line_part_time = time.time()
                if self.b_ready_second_process == False:
                    self.b_ready_second_process = True

                not_real_bbox = True

                self.l_lines, self.r_lines, self.left_fit_line, self.right_fit_line=LineDetector(frame_buffer[self.input_idx], height, width, self.l_lines, self.r_lines, self.left_fit_line, self.right_fit_line)

                if self.buf_idx == 0:
                    center_box_0, center_pt1_0, center_pt2_0, not_real_bbox, l_lines, r_lines, left_fit_line, right_fit_line, self.center_x, self.center_y = center_box_decider(frame_buffer[self.input_idx], self.l_lines, self.r_lines, self.left_fit_line, self.right_fit_line, not_real_bbox, self.center_x, self.center_y)
                elif self.buf_idx == 1:
                    center_box_1, center_pt1_1, center_pt2_1, not_real_bbox, self.l_lines, self.r_lines, self.left_fit_line, self.right_fit_line, self.center_x, self.center_y = center_box_decider(frame_buffer[self.input_idx], self.l_lines, self.r_lines, self.left_fit_line, self.right_fit_line, not_real_bbox, self.center_x, self.center_y)
                else:
                    center_box_2, center_pt1_2, center_pt2_2, not_real_bbox, self.l_lines, self.r_lines, self.left_fit_line, self.right_fit_line, self.center_x, self.center_y = center_box_decider(frame_buffer[self.input_idx], self.l_lines, self.r_lines, self.left_fit_line, self.right_fit_line, not_real_bbox, self.center_x, self.center_y)

                line_part_time = time.time() - line_part_time
                line_time += line_part_time

                sem[self.buf_idx].release()
            
            elif self.tid == 1:
                if self.b_ready_second_process == False: 
                    time.sleep(0.001)
                
                sem[self.buf_idx].acquire()
                if self.b_ready_third_process == False:
                    self.b_ready_third_process = True
                yolo_part_time = time.time()
                iou = []
                bbox_list = list()


                if self.buf_idx == 0:
                    self.cur_bbox, yolo_fail_0 = yolo_detect(frame_buffer[self.input_idx], center_box_0, yolo_fail_0, self.car_buffer)
                    self.alarmCnt = alarm(prev_bbox_0, self.cur_bbox, self.alarmCnt)
                    frame_buffer[self.input_idx] = draw_and_show(frame_buffer[self.input_idx], self.input_idx, center_pt1_0, center_pt2_0, self.cur_bbox, self.alarmCnt)
                    prev_bbox_0 = self.cur_bbox

                elif self.buf_idx == 1:
                    self.cur_bbox, yolo_fail_1 = yolo_detect(frame_buffer[self.input_idx], center_box_1, yolo_fail_1, self.car_buffer)
                    self.alarmCnt = alarm(prev_bbox_1, self.cur_bbox, self.alarmCnt)
                    frame_buffer[self.input_idx] = draw_and_show(frame_buffer[self.input_idx], self.input_idx, center_pt1_1, center_pt2_1, self.cur_bbox, self.alarmCnt)
                    prev_bbox_1 = self.cur_bbox

                else:
                    self.cur_bbox, yolo_fail_2 = yolo_detect(frame_buffer[self.input_idx], center_box_2, yolo_fail_2, self.car_buffer)
                    self.alarmCnt = alarm(prev_bbox_2, self.cur_bbox, self.alarmCnt)
                    frame_buffer[self.input_idx] = draw_and_show(frame_buffer[self.input_idx], self.input_idx, center_pt1_2, center_pt2_2, self.cur_bbox, self.alarmCnt)
                    prev_bbox2 = self.cur_bbox


                yolo_part_time = time.time() - yolo_part_time

                yolo_time += yolo_part_time
                sem[self.buf_idx].release()

            else:
                if self.b_ready_second_process == False:
                    time.sleep(0.002)
                if self.b_ready_third_process == False:
                    time.sleep(0.002)
                sem[self.buf_idx].acquire()
                tracking_part_time = time.time()
                if self.buf_idx == 0:
                    if prev_bbox_0 != [0,0,0,0] and yolo_fail_0 == True:
                        init_rect=[prev_bbox_0[0], prev_bbox_0[1], prev_bbox_0[2]-prev_bbox_0[0], prev_bbox_0[3]-prev_bbox_0[1]]
                        self.tracker.init(frame_buffer[self.input_idx],init_rect)
                        outputs = self.tracker.track(frame_buffer[self.input_idx])

                        tracking_bbox = list(map(float, outputs['bbox']))
                            
                        self.cur_bbox=(tracking_bbox[0], tracking_bbox[1],
                                  tracking_bbox[2]+tracking_bbox[0],
                                  tracking_bbox[3]+tracking_bbox[1])
                        self.alarmCnt = alarm(prev_bbox_0, self.cur_bbox, self.alarmCnt)
                        frame_buffer[self.input_idx] = draw_and_show(frame_buffer[self.input_idx], self.input_idx, center_pt1_0, center_pt2_0, self.cur_bbox, self.alarmCnt)
                        prev_bbox_0 = self.cur_bbox

                elif self.buf_idx == 1:
                    if prev_bbox_1 != [0,0,0,0] and yolo_fail_1 == True:
                        init_rect=[prev_bbox_1[0], prev_bbox_1[1], prev_bbox_1[2]-prev_bbox_1[0], prev_bbox_1[3]-prev_bbox_1[1]]
                        self.tracker.init(frame_buffer[self.input_idx],init_rect)
                        outputs = self.tracker.track(frame_buffer[self.input_idx])

                        tracking_bbox = list(map(float, outputs['bbox']))
                            
                        self.cur_bbox=(tracking_bbox[0], tracking_bbox[1],
                                  tracking_bbox[2]+tracking_bbox[0],
                                  tracking_bbox[3]+tracking_bbox[1])
                        self.alarmCnt = alarm(prev_bbox_1, self.cur_bbox, self.alarmCnt)
                        frame_buffer[self.input_idx] = draw_and_show(frame_buffer[self.input_idx], self.input_idx, center_pt1_1, center_pt2_1, self.cur_bbox, self.alarmCnt)
                        prev_bbox_1 = self.cur_bbox

                else:
                    if prev_bbox_2 != [0,0,0,0] and yolo_fail_2 == True:
                        init_rect=[prev_bbox_0[0], prev_bbox_0[1], prev_bbox_0[2]-prev_bbox_0[0], prev_bbox_0[3]-prev_bbox_0[1]]
                        self.tracker.init(frame_buffer[self.input_idx],init_rect)
                        outputs = self.tracker.track(frame_buffer[self.input_idx])

                        tracking_bbox = list(map(float, outputs['bbox']))
                            
                        self.cur_bbox=(tracking_bbox[0], tracking_bbox[1],
                                  tracking_bbox[2]+tracking_bbox[0],
                                  tracking_bbox[3]+tracking_bbox[1])
                        self.alarmCnt = alarm(prev_bbox_2, self.cur_bbox, self.alarmCnt)
                        frame_buffer[self.input_idx] = draw_and_show(frame_buffer[self.input_idx], self.input_idx, center_pt1_2, center_pt2_2, self.cur_bbox, self.alarmCnt)
                        prev_bbox2 = self.cur_bbox

                tracking_part_time = time.time() - tracking_part_time
                tracking_time += tracking_part_time

                sem[self.buf_idx].release()

            self.input_idx += 1
            self.buf_idx = (self.buf_idx+1) % NUM_BUFFER



if __name__ == "__main__":
    #global total_frame, width, height, line_time, yolo_time, tracking_time, frame_buffer
    
    YOLO()
    tracker = PYSOTINIT()
    
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    for vid in vidList:
        print("load" + str(vid))
        imgList = glob(vid + "/*.jpg")
        imgList.sort()
        for img in imgList:
            im = cv2.imread(img)
            frame_buffer.append(im)
        height, width = frame_buffer[0].shape[:2]
        total_frame = len(frame_buffer)
        
        line_time = 0
        yolo_time = 0
        tracking_time = 0

        thd_0 = MyThread(0, darknet_image, tracker)
        thd_1 = MyThread(1, darknet_image, tracker)
        thd_2 = MyThread(2, darknet_image, tracker)
        
        thd_0.start()
        thd_1.start()
        thd_2.start()

        thd_0.join()
        thd_1.join()
        thd_2.join()

        print("all time = " + str(line_time*1000 + yolo_time*1000 + tracking_time*1000) + "ms")
        print("all line detection time = " + str(line_time*1000) + "ms")
        print("all yolo detection time = " + str(yolo_time*1000) + "ms")
        print("all tracking time = " + str(tracking_time*1000) + "ms")

        print("\n" + "avg line detection time = " + str(line_time*1000/total_frame) + "ms")
        print("avg yolo detection time = " + str(yolo_time*1000/total_frame) + "ms")
        print("avg tracking time = " + str(tracking_time*1000/total_frame) + "ms")

        print("\n" + "pipe line time = " + str(yolo_time*1000) + "ms")
        print("avg time per frame = " + str(yolo_time*1000/total_frame) + "ms")
        print("avg fps = " + str(total_frame/yolo_time) + "fps")
