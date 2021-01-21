#!/usr/bin/env python

'''
Contain functions to draw Bird Eye View for region of interest(ROI) and draw bounding boxes according to risk factor
for humans in a frame and draw lines between boxes according to risk factor between two humans. 
'''

__title__           = "plot.py"
__Version__         = "1.0"
__copyright__       = "Copyright 2020 , Social Distancing AI"
__license__         = "MIT"
__author__          = "Deepak Birla"
__email__           = "birla.deepak26@gmail.com"
__date__            = "2020/05/29"
__python_version__  = "3.5.2"

# imports
import cv2
import numpy as np
from math import cos, sin
import math

class Plot:
    # Function to draw Bird Eye View for region of interest(ROI). Red, Yellow, Green points represents risk to human. 
    # Red: High Risk
    # Yellow: Low Risk
    # Green: No Risk
    def bird_eye_view(self, frame, distances_mat, bottom_points, scale_w, scale_h, risk_count, eye_points, rotations, plane_height):
        h = frame.shape[0]
        w = frame.shape[1]
        global saveEyePoints
        saveEyePoints = []
        red = (0, 0, 255)
        green = (0, 255, 0)
        yellow = (0, 255, 255)
        white = (200, 200, 200)

        blank_image = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
        blank_image[:] = white
        warped_pts = []
        r = []
        g = []
        y = []
        for i in range(len(distances_mat)):

            if distances_mat[i][2] == 0:
                if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                    r.append(distances_mat[i][0])
                if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                    r.append(distances_mat[i][1])

                blank_image = cv2.line(blank_image, (int(distances_mat[i][0][0] * scale_w), int(distances_mat[i][0][1] * scale_h)), (int(distances_mat[i][1][0] * scale_w), int(distances_mat[i][1][1]* scale_h)), red, 2)
                
        for i in range(len(distances_mat)):
                    
            if distances_mat[i][2] == 1:
                if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                    y.append(distances_mat[i][0])
                if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                    y.append(distances_mat[i][1])
            
                blank_image = cv2.line(blank_image, (int(distances_mat[i][0][0] * scale_w), int(distances_mat[i][0][1] * scale_h)), (int(distances_mat[i][1][0] * scale_w), int(distances_mat[i][1][1]* scale_h)), yellow, 2)
                
        for i in range(len(distances_mat)):
            
            if distances_mat[i][2] == 2:
                if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                    g.append(distances_mat[i][0])
                if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                    g.append(distances_mat[i][1])

        # Draw eye point line
        for count in range(0, len(eye_points)):
            blank_image = cv2.line(blank_image, tuple((int(eye_points[count][0] * scale_w), int(eye_points[count][1] * scale_h))),tuple((int(bottom_points[count][0] * scale_w), int(bottom_points[count][1] * scale_h))), [0,0,0], 3)
        count = 0
        for i in bottom_points:
            blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, yellow, 10)
            self.draw_axis(blank_image, rotations[count], tdx = int(i[0]  * scale_w), tdy = int(i[1] * scale_h), size = 30)
            self.draw_eye_direction(blank_image, rotations[count],[int(i[0]  * scale_w),int(i[1] * scale_h)], plane_height, int(h * scale_h), int(w * scale_w))
            count += 1


        # for i in y:
        #     blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, yellow, 10)
        # for i in r:
        #     blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, red, 10)
            
        #pad = np.full((100,blank_image.shape[1],3), [110, 110, 100], dtype=np.uint8)
        #cv2.putText(pad, "-- HIGH RISK : " + str(risk_count[0]) + " people", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        #cv2.putText(pad, "-- LOW RISK : " + str(risk_count[1]) + " people", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        #cv2.putText(pad, "-- SAFE : " + str(risk_count[2]) + " people", (50,  80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        #blank_image = np.vstack((blank_image,pad))   
            
        return blank_image

    def getIntersec(self, rayPoint, rayDirection, bottom_y, right_x, planeSide = 'floor' , epsilon=1e-6):
        # Define plane
        temp = 'floor'
        if planeSide == 'floor':
            temp = 'floor'
            planeNormal = np.array([0, 0, 1])
            planePoint = np.array([0, 0, 0])  # Any point on the plane planePoint(planeNormal)
        elif planeSide == 'front':
            temp = 'front'
            planeNormal = np.array([0, 1, 0])
            planePoint = np.array([0, bottom_y, 0])  # Any point on the plane planePoint(planeNormal)
        elif planeSide == 'left':
            temp = 'left'
            planeNormal = np.array([1, 0, 0])
            planePoint = np.array([0, 0, 0])  # Any point on the plane planePoint(planeNormal)
        else:
            #### right wall
            temp = 'right'
            planeNormal = np.array([-1, 0, 0])
            planePoint = np.array([right_x, 0, 0])  # Any point on the plane planePoint(planeNormal)
        # Define ray
        # rayDirection = np.array([0, 0.5, -0.5])
        # rayPoint = np.array([0, 0, 1]) #Any point along the ray rayPoint(rayDirection)

        #### line-plane collision
        ndotu = planeNormal.dot(rayDirection)
        if abs(ndotu) < epsilon:
            #### create front room wall
            planeNormal = np.array([0, 1, 0])
            planePoint = np.array([0, bottom_y, 0])  # Any point on the plane planePoint(planeNormal)
            ndotu = planeNormal.dot(rayDirection)
            temp = 'front'
            # raise RuntimeError("no intersection or line is within plane")
            if abs(ndotu) < epsilon:
                #### create right room wall
                planeNormal = np.array([-1, 0, 0])
                planePoint = np.array([right_x, 0, 0])  # Any point on the plane planePoint(planeNormal)
                ndotu = planeNormal.dot(rayDirection)
                temp = 'right'
                if abs(ndotu) < epsilon:
                    #### create left room wall
                    planeNormal = np.array([1, 0, 0])
                    planePoint = np.array([0, 0, 0])  # Any point on the plane planePoint(planeNormal)
                    ndotu = planeNormal.dot(rayDirection)
                    temp = 'left'
                    if abs(ndotu) < epsilon:
                        raise RuntimeError("no intersection or line is within plane")
        # print(temp)
        w = rayPoint - planePoint
        si = -planeNormal.dot(w) / ndotu
        Psi = w + si * rayDirection + planePoint
        if temp == 'front':
            Psi[1] = bottom_y
        elif temp == 'left':
            Psi[0] = 0
        elif temp == 'right':
            Psi[0] = right_x
        return Psi

    def getHeadPoseRayDirection(self, rotation, bottom_point, plane_height):
        # print(rotation)
        yaw, pitch, roll = rotation
        yaw = math.radians(yaw)
        # pitch = np.clip(pitch,-125,0)
        pitch = math.radians(pitch)
        roll = math.radians(roll)
        #### x axis
        # x = cos(yaw) * cos(pitch)
        # y = sin(yaw) * cos(pitch)
        # z = sin(pitch)
        #### y axis
        x = -cos(yaw) * sin(pitch) * sin(roll) - sin(yaw) * cos(roll)
        y = -sin(yaw) * sin(pitch) * sin(roll) + cos(yaw) * cos(roll)
        z = cos(pitch) * sin(roll)
        # print([x,y,z])
        return np.array([x, y, z])

    def draw_eye_direction(self, img, rotation, bottom_point, plane_height, bottom_y, right_x):
        red = (0, 0, 255)
        green = (0, 255, 0)
        yaw, pitch, roll = rotation
        headPoint = [bottom_point[0],bottom_point[1],plane_height]
        headDirection = self.getHeadPoseRayDirection(rotation, bottom_point, plane_height)
        # print(headDirection)
        cv2.circle(img, (int(headDirection[0]), int(headDirection[1])), 1, (0,0,0), 1)
        # floor check collision
        eyePoint = self.getIntersec(headPoint, headDirection, bottom_y, right_x)
        #if (eyePoint[1] < 0 or eyePoint[1] > bottom_y) and (eyePoint[0]>0 and eyePoint[0]<right_x): # old
        if eyePoint[1]< 0 or eyePoint[1]> bottom_y:
            eyePoint = self.getIntersec(headPoint, headDirection, bottom_y, right_x, 'front')
        # เกินขอบ front
        if eyePoint[0]< 0:
            eyePoint = self.getIntersec(headPoint, headDirection, bottom_y, right_x, 'left')
        elif eyePoint[0]> right_x:
            eyePoint = self.getIntersec(headPoint, headDirection, bottom_y, right_x, 'right')

        #### ถ้าเป็นมุมป้าน
        if eyePoint[1] <= headPoint[1]:
            eyePoint = self.getIntersec(headPoint, headDirection, bottom_y, right_x, 'front')
            if eyePoint[0] < 0:
                eyePoint = self.getIntersec(headPoint, headDirection, bottom_y, right_x, 'left')
            elif eyePoint[0] > right_x:
                eyePoint = self.getIntersec(headPoint, headDirection, bottom_y, right_x, 'right')

        if pitch >= -125:
            cv2.circle(img, (int(eyePoint[0]), int(eyePoint[1])), 2, green, 10)
        else:
            cv2.circle(img, (int(eyePoint[0]),int(eyePoint[1])), 2, red, 10)
        cv2.line(img, tuple(bottom_point), (int(eyePoint[0]), int(eyePoint[1])), (0,0,0))
        # cv2.putText(img, str(pitch), (int(eyePoint[0]),int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        saveEyePoints.append(eyePoint.tolist())
        return img

    def getEyePoints(self):
        return saveEyePoints

    ## draw x y z
    def draw_axis(self, img, rotation, tdx=None, tdy=None, size = 100):

        red = (0, 0, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)
        yellow = (0, 255, 255)

        yaw, pitch, roll = rotation
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2

        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        # size *= 10
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)), red, 3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)), green, 3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)), blue, 2)

        return img

    # Function to draw bounding boxes according to risk factor for humans in a frame and draw lines between
    # boxes according to risk factor between two humans.
    # Red: High Risk
    # Yellow: Low Risk
    # Green: No Risk 
    def social_distancing_view(self, frame, distances_mat, boxes, risk_count):
        
        red = (0, 0, 255)
        green = (0, 255, 0)
        yellow = (0, 255, 255)
        
        for i in range(len(boxes)):

            x,y,w,h = boxes[i][:]
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),green,2)
                            
        for i in range(len(distances_mat)):

            per1 = distances_mat[i][0]
            per2 = distances_mat[i][1]
            closeness = distances_mat[i][2]
            
            if closeness == 1:
                x,y,w,h = per1[:]
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),yellow,2)
                    
                x1,y1,w1,h1 = per2[:]
                frame = cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),yellow,2)
                    
                frame = cv2.line(frame, (int(x+w/2), int(y+h/2)), (int(x1+w1/2), int(y1+h1/2)),yellow, 2)
                
        for i in range(len(distances_mat)):

            per1 = distances_mat[i][0]
            per2 = distances_mat[i][1]
            closeness = distances_mat[i][2]
            
            if closeness == 0:
                x,y,w,h = per1[:]
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),red,2)
                    
                x1,y1,w1,h1 = per2[:]
                frame = cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),red,2)
                    
                frame = cv2.line(frame, (int(x+w/2), int(y+h/2)), (int(x1+w1/2), int(y1+h1/2)),red, 2)
                
        pad = np.full((140,frame.shape[1],3), [110, 110, 100], dtype=np.uint8)
        # cv2.putText(pad, "Bounding box shows the level of risk to the person.", (50, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 0), 2)
        # cv2.putText(pad, "-- HIGH RISK : " + str(risk_count[0]) + " people", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        # cv2.putText(pad, "-- LOW RISK : " + str(risk_count[1]) + " people", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        # cv2.putText(pad, "-- SAFE : " + str(risk_count[2]) + " people", (50,  100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        frame = np.vstack((frame,pad))
                
        return frame

