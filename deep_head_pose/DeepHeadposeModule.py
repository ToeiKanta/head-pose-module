import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import hopenet, utils

def get_project_dir():
    current_path = os.path.abspath(os.path.join(__file__, "../"))
    return current_path

def relative(path):
    path = os.path.join(get_project_dir(), path)
    return os.path.abspath(path)

class DeepHeadposeModule:

    def __init__(
        self,
        gpu=1,
        snapshot_path = relative('files/hopenet_robust_alpha1.pkl'),
    ):
        cudnn.enabled = True
        self.gpu = 1
        # ResNet50 structure
        self.model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        print('Loading snapshot.')
        # Load snapshot
        saved_state_dict = torch.load(snapshot_path)
        self.model.load_state_dict(saved_state_dict)
        print('Loading data.')
        self.transformations = transforms.Compose([transforms.Scale(224),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.model.cuda(gpu)
        print('Ready to test network.')
        # Test the Model
        self.model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).cuda(gpu)

    def getPose(self, frame, box): # frame is full image, box from RetinaFace
        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # Get x_min, y_min, x_max, y_max from RetinaFace's Box
        x_min = box[0]#-(box[2]-box[0])/5
        y_min = box[1]#-(box[3]-box[1])/2
        x_max = box[2]#+(box[2]-box[0])/5
        y_max = box[3]#+(box[3]-box[1])/2
        bbox_width = abs(x_max - x_min)
        bbox_height = abs(y_max - y_min)
        x_min -= 2 * bbox_width / 4
        x_max += 2 * bbox_width / 4
        y_min -= 3 * bbox_height / 4
        y_max += bbox_height / 4
        x_min = max(x_min, 0); y_min = max(y_min, 0)
        x_max = min(frame.shape[1], x_max); y_max = min(frame.shape[0], y_max)
        # Crop image
        img = cv2_frame[int(y_min):int(y_max),int(x_min):int(x_max)]
        img = Image.fromarray(img)

        # Transform
        img = self.transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img).cuda(gpu)

        yaw, pitch, roll = self.model(img)

        yaw_predicted = F.softmax(yaw)
        pitch_predicted = F.softmax(pitch)
        roll_predicted = F.softmax(roll)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99

        # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
        utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
        # Plot expanded bounding box
        # cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,255,0), 1)
        return yaw_predicted,pitch_predicted,roll_predicted, frame
