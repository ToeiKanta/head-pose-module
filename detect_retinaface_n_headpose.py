import cv2
import numpy as np
import os
import time
import pickle
import headpose
import argparse

from face_detection import RetinaFace
filename = 'Class2.png'
#model = 'resnet50'
model = 'mobilenet0.25'
name = 'retinaFace'
scale = 0.5
raw_img = cv2.imread(os.path.join('./Test', filename))
CONFIDENCE = 0.1
count = 0
width, height = raw_img.shape[:2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', metavar='FILE', dest='input_file', default=None, help='Input video. If not given, web camera will be used.')
    parser.add_argument('-o', metavar='FILE', dest='output_file', default=None, help='Output video.')
    parser.add_argument('-wh', metavar='N', dest='wh', default=[720, 480], nargs=2, help='Frame size.')
    parser.add_argument('-lt', metavar='N', dest='landmark_type', type=int, default=1, help='Landmark type.')
    parser.add_argument('-lp', metavar='FILE', dest='landmark_predictor', 
                        default='model/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
    args = vars(parser.parse_args())

    detector = RetinaFace(gpu_id=-1)
    t0 = time.time()
    print('start')
    # resize image
    print(f'w: {width*scale} h: {height*scale}')
    raw_img = cv2.resize(raw_img, (int(height*scale) , int(width*scale)))
    faces = detector(raw_img)
    t1 = time.time()
    print(f'took {round(t1 - t0, 3)} to get {len(faces)} box')
    # Initialize head pose detection
    hpd = headpose.HeadposeDetection(args["landmark_type"], args["landmark_predictor"])

    
    for box, landmarks, score in faces:
        box = box.astype(np.int)
        if score < CONFIDENCE:
            continue
        cropped = raw_img[box[1]:box[3], box[0]:box[2]]
        newimg = cv2.resize(cropped, (112, 112))
        # cv2.imwrite("../cropped_face/face_" + str(count) + ".jpg", newimg)
        
        #head-pose
        # frame = cv2.flip(raw_img, 1)
        frame, angles = hpd.process_image(raw_img)
        # frame, angles = hpd.process_image(cropped)
        # Display the resulting frame
        cv2.imshow('IMG', frame)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                headpose.t.summary()
                break
        count+=1
        cv2.rectangle(raw_img, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=1)

    font = cv2.FONT_HERSHEY_DUPLEX
    text = f'took {round(t1 - t0, 3)} to get {count} faces'
    cv2.putText(raw_img, text, (20, 20), font, 0.5, (255, 255, 255), 1)
    cv2.imwrite(os.path.join('./output', f'{name}_{model}_{scale}_{filename}'), raw_img)

    while True:
        cv2.imshow('IMG', raw_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

