#
#   Headpose Detection Utils
#   Written by Qhan
#   Last Update: 2019.1.9
#

import numpy as np
import cv2
from math import cos, sin
import math

class Color():
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


class Annotator():
    ## rvec = rotation_vector
    ## tvec = translation_vector
    def __init__(self, im, angles=None, bbox=None, lm=None, rvec=None, tvec=None, cm=None, dc=None, b=10.0):
        self.im = im

        self.angles = angles
        self.bbox = bbox
        self.lm = lm
        self.rvec = rvec
        self.tvec = tvec
        self.cm = cm
        self.dc = dc
        self.nose = tuple(lm[0].astype(int))
        self.box = np.array([
            ( b,  b,  b), ( b,  b, -b), ( b, -b, -b), ( b, -b,  b),
            (-b,  b,  b), (-b,  b, -b), (-b, -b, -b), (-b, -b,  b)
        ])
        self.b = b

        h, w, c = im.shape
        self.fs = ((h + w) / 2) / 500
        self.ls = round(self.fs * 2)
        self.ls = 1
        self.ps = self.ls


    def draw_all(self,howLong):
        # self.draw_bbox()
        # self.draw_landmarks()
        backBoxCenter, frontBoxCenter = self.draw_axes()
        # self.draw_direction_axes(backBoxCenter, frontBoxCenter)
        eye_direction_point = self.draw_direction(howLong)
        # self.draw_info()
        # self.draw_direction_2()
        return self.im, eye_direction_point

    def get_image(self):
        return self.im


    def draw_bbox(self):
        x1, y1, x2, y2 = np.array(self.bbox).astype(int)
        cv2.rectangle(self.im, (x1, y1), (x2, y2), Color.green, self.ls)


    def draw_landmarks(self):
        for p in self.lm:
            point = tuple(p.astype(int))
            cv2.circle(self.im, point, self.ps, Color.red, -1)


    # axis lines index
    box_lines = np.array([
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ])
    def draw_axes(self):
        (projected_box, _) = cv2.projectPoints(self.box, self.rvec, self.tvec, self.cm, self.dc)
        pbox = projected_box[:, 0]
        for p in self.box_lines:
            p1 = tuple(pbox[p[0]].astype(int))
            p2 = tuple(pbox[p[1]].astype(int))
            if (p[0], p[1]) == (0, 1) or \
                (p[0], p[1]) == (4, 5) or \
                    (p[0], p[1]) == (6, 7) or \
                    (p[0], p[1]) == (2, 3):
                cv2.line(self.im, p1, p2, Color.red, self.ls)
            elif (p[0], p[1]) == (1, 2):
                cv2.line(self.im, p1, p2, Color.yellow, self.ls)
            else:
                cv2.line(self.im, p1, p2, Color.blue, self.ls)
        backBoxCenter = (int((pbox[1][0] + pbox[5][0])/2), int((pbox[1][1]+pbox[2][1])/2))
        frontBoxCenter = (int((pbox[0][0] + pbox[4][0]) / 2), int((pbox[0][1] + pbox[3][1]) / 2))
        return backBoxCenter, frontBoxCenter

    def draw_axis(self, tdx=None, tdy=None, size=150.):
        x1, y1, x2, y2 = np.array(self.bbox).astype(int)
        size = (x2-x1)/2
        img = self.im
        yaw,pitch,roll = self.angles # x,y,z => yaw,pitch,roll
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
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    def draw_direction_axes(self, backBoxCenter , frontBoxCenter,howLong = 50):
        c = [0, 0]
        lenAB = math.sqrt(math.pow(backBoxCenter[0] - frontBoxCenter[0], 2.0) + math.pow(backBoxCenter[1] - frontBoxCenter[1], 2.0))
        try:
            c[0] = int (frontBoxCenter[0]+(frontBoxCenter[0] - backBoxCenter[0]) / lenAB * howLong)
            c[1] = int (frontBoxCenter[1]+(frontBoxCenter[1] - backBoxCenter[1]) / lenAB * howLong)
            cv2.line(self.im, backBoxCenter, tuple(c), Color.yellow, self.ls)
            # cv2.line(self.im, backBoxCenter, frontBoxCenter, Color.green, self.ls*2)
        except:
            return

    def draw_direction(self, howLong = 100):
        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, self.b)]), self.rvec, self.tvec, self.cm, self.dc)
        p1 = self.nose
        p2 = tuple(nose_end_point2D[0, 0].astype(int))
        # delete hard error (only line drawing)
        # if p2[0] <= 0 or p2[1] <= 0:
        #     return
        # elif p2[0] >= self.im.shape[0] or p2[1] >= self.im.shape[1]:
        #     return
        # extend line longer
        c = [0,0]
        temp = [0, 0] # temp is eye direction point (used in Bird Eye View)
        lenAB = math.sqrt(math.pow(p1[0] - p2[0], 2.0) + math.pow(p1[1] - p2[1], 2.0))
        try:
            c[0] = int (p2[0]+(p2[0] - p1[0]) / lenAB * howLong)
            c[1] = int (p2[1]+(p2[1] - p1[1]) / lenAB * howLong)
            temp[0] = c[0]
            temp[1] = c[1] + int(self.bbox[3] - p1[1])
            # print(f"p1: {p1}")
            # cv2.line(self.im, p1, tuple(c), Color.yellow, self.ls)
            # cv2.line(self.im, p1, p2, Color.green, self.ls)
            return tuple(temp)
        except:
            temp[0] = p2[0]
            temp[1] = p2[1] + int(self.bbox[3] - p1[1])
            return tuple(temp)

    def get_direction_point(self, howLong = 100):
        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, self.b)]), self.rvec, self.tvec, self.cm, self.dc)
        p1 = self.nose
        p2 = tuple(nose_end_point2D[0, 0].astype(int))
        # delete hard error (only line drawing)
        # if p2[0] <= 0 or p2[1] <= 0:
        #     return
        # elif p2[0] >= self.im.shape[0] or p2[1] >= self.im.shape[1]:
        #     return
        # extend line longer
        c = [0,0]
        temp = [0, 0]
        temp[0] = int(p1[0])
        temp[1] = int(p1[1] + (self.bbox[3] - p1[1]))
        lenAB = math.sqrt(math.pow(p1[0] - p2[0], 2.0) + math.pow(p1[1] - p2[1], 2.0))
        try:
            c[0] = int (p2[0]+(p2[0] - p1[0]) / lenAB * howLong)
            c[1] = int ((p2[1]+(p2[1] - p1[1]) / lenAB * howLong) + (self.bbox[3] - p1[1])) # ขยับ y จากคาง

            return tuple(c)
        except:
            return p2

    def draw_direction_2(self):
        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, self.b)]), self.rvec, self.tvec, self.cm, self.dc)
        p1 = self.nose
        p2 = tuple(nose_end_point2D[0, 0].astype(int))
        self.draw_axis(tdx=p1[0],tdy=p1[1],size=40)

    def draw_info(self, fontColor=Color.yellow):
        x, y, z = self.angles
        px, py, dy = int(5 * self.fs), int(25 * self.fs), int(30 * self.fs)
        font = cv2.FONT_HERSHEY_DUPLEX
        fs = self.fs
        cv2.putText(self.im, "X: %+06.2f" % x, (px, py), font, fontScale=fs, color=fontColor)
        cv2.putText(self.im, "Y: %+06.2f" % y, (px, py + dy), font, fontScale=fs, color=fontColor)
        cv2.putText(self.im, "Z: %+06.2f" % z, (px, py + 2 * dy), font, fontScale=fs, color=fontColor)
