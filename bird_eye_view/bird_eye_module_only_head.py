#!/usr/bin/env python

'''
Calculates Region of Interest(ROI) by receiving points from mouse event and transform prespective so that
we can have top view of scene or ROI. This top view or bird eye view has the property that points are
distributed uniformally horizontally and vertically(scale for horizontal and vertical direction will be
 different). So for bird eye view points are equally distributed, which was not case for normal view.

YOLO V3 is used to detect humans in frame and by calculating bottom center point of bounding boxe around humans, 
we transform those points to bird eye view. And then calculates risk factor by calculating distance between
points and then drawing birds eye view and drawing bounding boxes and distance lines between boxes on frame.
'''

__title__           = "main.py"
__Version__         = "1.0"
__copyright__       = "Copyright 2020 , Social Distancing AI"
__license__         = "MIT"
__author__          = "Deepak Birla"
__email__           = "birla.deepak26@gmail.com"
__date__            = "2020/05/29"
__python_version__  = "3.5.2"

# imports
import codecs
import json

import cv2
import numpy as np
import time
import argparse
import math
# own modules
from .utills import Utills
from .plot import Plot
import pickle
import os
confid = 0.5
thresh = 0.5
mouse_pts = []
points = []


def get_project_dir():
    current_path = os.path.abspath(os.path.join(__file__, "../"))
    return current_path

def relative(path):
    path = os.path.join(get_project_dir(), path)
    return os.path.abspath(path)

class BirdEyeModuleOnlyHead:
    def __init__(self, video_path, output_dir, output_vid, opencv, scale, closeImShow, bird_width, bird_height, plane_height):
        # global cv2
        global utills
        global plot
        self.bird_width = bird_width
        self.bird_height = bird_height
        print("\nbird_width: " + str(bird_width) + " bird_height: " + str(bird_height));
        print("\nplane_height: " + str(plane_height));
        self.plane_height = plane_height;
        self.closeImShow = closeImShow
        plot = Plot()
        utills = Utills(bird_width, bird_height)
        self.savePath = relative('save_files/' + os.path.basename(video_path) + '-' + str(scale) + '.pk')
        # cv2 = opencv

        # Receives arguements specified by user
        # parser = argparse.ArgumentParser()
        
        # parser.add_argument('-v', '--video_path', action='store', dest='video_path', default='./data/example.mp4' ,
        #                 help='Path for input video')
                        
        # parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/' ,
        #                 help='Path for Output images')
        
        # parser.add_argument('-O', '--output_vid', action='store', dest='output_vid', default='./output_vid/' ,
        #                 help='Path for Output videos')

        # parser.add_argument('-m', '--model', action='store', dest='model', default='./models/',
        #                 help='Path for models directory')
                        
        # parser.add_argument('-u', '--uop', action='store', dest='uop', default='NO',
        #                 help='Use open pose or not (YES/NO)')
                        
        # values = parser.parse_args()
        model = './models/'
        model_path = model
        if model_path[len(model_path) - 1] != '/':
            model_path = model_path + '/'
            
        if output_dir[len(output_dir) - 1] != '/':
            self.output_dir = output_dir + '/'
        
        if output_vid[len(output_vid) - 1] != '/':
            self.output_vid = output_vid + '/'

        if os.path.exists(self.savePath):
            global mouse_pts;
            mouse_pts = pickle.load(open(self.savePath, 'rb'))
            self.firstSetup = False
        else:
            self.firstSetup = True
            print("First Setup Bird View.")
            # set mouse callback 
            cv2.namedWindow("imageBird")
            cv2.setMouseCallback("imageBird", self.get_mouse_points)
            np.random.seed(42)
        # load Yolov3 weights
        
        # weightsPath = model_path + "yolov3.weights"
        # configPath = model_path + "yolov3.cfg"

        # net_yl = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        # ln = net_yl.getLayerNames()
        # ln1 = [ln[i[0] - 1] for i in net_yl.getUnconnectedOutLayers()]

        
        # self.calculate_social_distancing(net_yl, ln1)
    # Function to get points for Region of Interest(ROI) and distance scale. It will take 8 points on first frame using mouse click    
    # event.First four points will define ROI where we want to moniter social distancing. Also these points should form parallel  
    # lines in real world if seen from above(birds eye view). Next 3 points will define 6 feet(unit length) distance in     
    # horizontal and vertical direction and those should form parallel lines with ROI. Unit length we can take based on choice.
    # Points should pe in pre-defined order - bottom-left, bottom-right, top-right, top-left, point 5 and 6 should form     
    # horizontal line and point 5 and 7 should form verticle line. Horizontal and vertical scale will be different. 

    # Function will be called on mouse events                                                          

    ## คำนวนองศาของระนาบ Yaw
    def getYawAdded(self):
        temp = self.ang((mouse_pts[0],mouse_pts[1]),((mouse_pts[0][0],mouse_pts[0][1]),(mouse_pts[1][0],mouse_pts[0][1])))
        return temp

    def dot(self, vA, vB):
        return vA[0]*vB[0]+vA[1]*vB[1]

    def ang(self, lineA, lineB):
        # Get nicer vector form
        vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
        vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
        # Get dot prod
        dot_prod = self.dot(vA, vB)
        # Get magnitudes
        magA = self.dot(vA, vA)**0.5
        magB = self.dot(vB, vB)**0.5
        # Get cosine value
        cos_ = dot_prod/magA/magB
        # Get angle in radians and then convert to degrees
        angle = math.acos(dot_prod/magB/magA)
        # Basically doing angle <- angle mod 360
        ang_deg = math.degrees(angle)%360

        if ang_deg-180>=0:
            # As in if statement
            return 360 - ang_deg
        else: 
            return ang_deg

    def get_mouse_points(self, event, x, y, flags, param):
        global mouse_pts
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(mouse_pts) < 4:
                cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
            else:
                cv2.circle(image, (x, y), 5, (255, 0, 0), 10)
                
            if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
                cv2.line(image, (x, y), (mouse_pts[len(mouse_pts)-1][0], mouse_pts[len(mouse_pts)-1][1]), (70, 70, 70), 2)
                if len(mouse_pts) == 3:
                    cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)
            
            if "mouse_pts" not in globals():
                mouse_pts = []
            mouse_pts.append((x, y))
            #print("Point detected")
            #print(mouse_pts)
            
    def calculate_social_distancing_retina_box(self, boxes, img, rotations):
        # Set scale for birds eye view
        # Bird's eye view will only show ROI
        global image
        image = img
        (height, width) = image.shape[:2]
        scale_w, scale_h = utills.get_scale(width, height)

        # fourcc = cv2.VideoWriter_fourcc(*"XVID")
        # output_movie = cv2.VideoWriter("./output_vid/distancing.avi", fourcc, fps, (width, height))
        # bird_movie = cv2.VideoWriter("./output_vid/bird_eye_view.avi", fourcc, fps, (int(width * scale_w), int(height * scale_h)))

        # (H, W) = frame.shape[:2]
        H = height
        W = width
        
        # first frame will be used to draw ROI and horizontal and vertical 180 cm distance(unit length in both directions)
        if self.firstSetup:
            print("first setup")
            print(f"\nbird eye (w,h): {width},{height}")
            while True:
                cv2.imshow("imageBird", image)
                cv2.waitKey(1)
                if len(mouse_pts) == 8:
                    cv2.destroyWindow("imageBird")
                    self.firstSetup = False
                    # if not os.path.exists(self.savePath):
                    #     with open(self.savePath, 'w'): 
                    #         pass;
                    pickle.dump(mouse_pts, open(self.savePath, 'wb'))
                    break
        points = mouse_pts

        # Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are
        # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view.
        # This bird eye view then has the property property that points are distributed uniformally horizontally and
        # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are
        # equally distributed, which was not case for normal view.
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        self.prespective_transform = cv2.getPerspectiveTransform(src, dst)

        # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, self.prespective_transform)[0]

        # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
        # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
        # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
        # which we can use to calculate distance between two humans in transformed view or bird eye view
        self.distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        self.distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)

        pnts = np.array(mouse_pts[:4], np.int32)
        cv2.polylines(image, [pnts], True, (70, 70, 70), thickness=2)
            
        # Here we will be using bottom center point of bounding box for all boxes and will transform all those
        # bottom center points to bird eye view
        # print(f"box: {boxes}")
        boxes1 = []
        for box in boxes:
            boxes1.append([int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])])
        person_points = utills.get_transformed_points(boxes1, self.prespective_transform)
        # eye_points = utills.get_transformed_eye_direct_point(person_points, self.prespective_transform, 300)
        eye_points = []
        # Here we will calculate distance between transformed points(humans)
        distances_mat, bxs_mat = utills.get_distances(boxes1, person_points, self.distance_w, self.distance_h)
        risk_count = utills.get_count(distances_mat)
    
        frame1 = np.copy(image)
        
        # Draw bird eye view and frame with bouding boxes around humans according to risk factor
        plot = Plot()
        bird_image = plot.bird_eye_view(image, distances_mat, person_points, scale_w, scale_h, risk_count, eye_points, rotations, self.plane_height)
        # img = plot.social_distancing_view(frame1, bxs_mat, boxes1, risk_count)

        person_point_scaled = []
        for person_point in person_points:
            person_point_scaled.append((person_point[0] * scale_w,person_point[1] * scale_h))
        # Show/write image and videos
        if not self.firstSetup:

            # numpy_horizontal = np.hstack((bird_image, img))
            # numpy_horizontal_concat = np.concatenate((bird_image, img), axis=1)
            # bird_movie.write(bird_image)
            if not self.closeImShow:
                cv2.imshow('Bird Eye View', bird_image)
                cv2.waitKey(1)
                cv2.imshow('Origin', img)
                cv2.waitKey(1)
            # cv2.imwrite(output_dir+"frame%d.jpg" % count, img)
            # cv2.imwrite(output_dir+"bird_eye_view/frame%d.jpg" % count, bird_image)
        # cv2.destroyAllWindows()
        
        return bird_image, plot.getEyePoints(), person_point_scaled

    # def calculate_social_distancing(self, net, ln1):
    #     vid_path = self.vid_path
    #     output_dir = self.output_dir
    #     output_vid = self.output_vid
    #
    #     count = 0
    #     vs = cv2.VideoCapture(vid_path)
    #
    #     # Get video height, width and fps
    #     height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     fps = int(vs.get(cv2.CAP_PROP_FPS))
    #
    #     # Set scale for birds eye view
    #     # Bird's eye view will only show ROI
    #     scale_w, scale_h = utills.get_scale(width, height)
    #
    #     fourcc = cv2.VideoWriter_fourcc(*"XVID")
    #     output_movie = cv2.VideoWriter("./output_vid/distancing.avi", fourcc, fps, (width, height))
    #     bird_movie = cv2.VideoWriter("./output_vid/bird_eye_view.avi", fourcc, fps, (int(width * scale_w), int(height * scale_h)))
    #
    #     points = []
    #     global image
    #
    #     while True:
    #
    #         (grabbed, frame) = vs.read()
    #
    #         if not grabbed:
    #             print('here')
    #             break
    #
    #         (H, W) = frame.shape[:2]
    #
    #         # first frame will be used to draw ROI and horizontal and vertical 180 cm distance(unit length in both directions)
    #         if count == 0:
    #             while True:
    #                 image = frame
    #                 cv2.imshow("imageBird", image)
    #                 cv2.waitKey(1)
    #                 if len(mouse_pts) == 8:
    #                     cv2.destroyWindow("imageBird")
    #                     break
    #
    #             points = mouse_pts
    #
    #         # Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are
    #         # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view.
    #         # This bird eye view then has the property property that points are distributed uniformally horizontally and
    #         # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are
    #         # equally distributed, which was not case for normal view.
    #         src = np.float32(np.array(points[:4]))
    #         dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
    #         prespective_transform = cv2.getPerspectiveTransform(src, dst)
    #
    #         # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
    #         pts = np.float32(np.array([points[4:7]]))
    #         warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]
    #
    #         # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
    #         # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
    #         # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
    #         # which we can use to calculate distance between two humans in transformed view or bird eye view
    #         distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
    #         distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
    #         pnts = np.array(points[:4], np.int32)
    #         cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)
    #
    #     ####################################################################################
    #
    #         # YOLO v3
    #         blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    #         net.setInput(blob)
    #         start = time.time()
    #         layerOutputs = net.forward(ln1)
    #         end = time.time()
    #         boxes = []
    #         confidences = []
    #         classIDs = []
    #
    #         for output in layerOutputs:
    #             for detection in output:
    #                 scores = detection[5:]
    #                 classID = np.argmax(scores)
    #                 confidence = scores[classID]
    #                 # detecting humans in frame
    #                 if classID == 0:
    #
    #                     if confidence > confid:
    #
    #                         box = detection[0:4] * np.array([W, H, W, H])
    #                         (centerX, centerY, width, height) = box.astype("int")
    #
    #                         x = int(centerX - (width / 2))
    #                         y = int(centerY - (height / 2))
    #
    #                         boxes.append([x, y, int(width), int(height)])
    #                         confidences.append(float(confidence))
    #                         classIDs.append(classID)
    #
    #         idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
    #         font = cv2.FONT_HERSHEY_PLAIN
    #         boxes1 = []
    #         for i in range(len(boxes)):
    #             if i in idxs:
    #                 boxes1.append(boxes[i])
    #                 x,y,w,h = boxes[i]
    #
    #         if len(boxes1) == 0:
    #             count = count + 1
    #             continue
    #
    #         # Here we will be using bottom center point of bounding box for all boxes and will transform all those
    #         # bottom center points to bird eye view
    #         person_points = utills.get_transformed_points(boxes1, prespective_transform)
    #
    #         # Here we will calculate distance between transformed points(humans)
    #         distances_mat, bxs_mat = utills.get_distances(boxes1, person_points, distance_w, distance_h)
    #         risk_count = utills.get_count(distances_mat)
    #
    #         frame1 = np.copy(frame)
    #
    #         # Draw bird eye view and frame with bouding boxes around humans according to risk factor
    #         bird_image = plot.bird_eye_view(frame, distances_mat, person_points, scale_w, scale_h, risk_count)
    #         img = plot.social_distancing_view(frame1, bxs_mat, boxes1, risk_count)
    #
    #         # Show/write image and videos
    #         if count != 0:
    #             output_movie.write(img)
    #
    #             # numpy_horizontal = np.hstack((bird_image, img))
    #             # numpy_horizontal_concat = np.concatenate((bird_image, img), axis=1)
    #             bird_movie.write(bird_image)
    #             cv2.imshow('Bird Eye View', bird_image)
    #             cv2.imshow('Origin', img)
    #             cv2.imwrite(output_dir+"frame%d.jpg" % count, img)
    #             cv2.imwrite(output_dir+"bird_eye_view/frame%d.jpg" % count, bird_image)
    #
    #         count = count + 1
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #
    #     vs.release()
    #     cv2.destroyAllWindows()



