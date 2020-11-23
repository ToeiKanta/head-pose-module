import cv2
import numpy as np
import dlib
import time
import pickle

class FaceRecognition_module:
    def __init__(self):
        # ---------- load face landmark predictor  --------------------------
        self.sp = dlib.shape_predictor('../facialLandmarks/shape_predictor_68_face_landmarks.dat')

        # ---------- load resnet model for recognition --------------------------
        self.model = dlib.face_recognition_model_v1('../faceRecognition/dlib_face_recognition_resnet_model_v1.dat')

        # ---------- load face bank  --------------------------
        self.FACE_DESC, self.FACE_NAME = pickle.load(open('../faceRecognition/tempmodel/trainset.pk', 'rb'))

        # ---------- read video --------------------------
        # cap = cv2.VideoCapture('0') #read from web camera
        self.cap = cv2.VideoCapture('../testVideo/test2.mp4')

        # ---------- write out video result -----------------
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.out = cv2.VideoWriter('output.mp4', fourcc, 15.0, (1920, 1080))
        self.cap.set(3, 1920)
        self.cap.set(4, 1080)
        self.face_detector = RetinaFace(gpu_id=0)
        print("load retina face done!!")

    def detect(self,frame, box, landmarks, score): #frame is (full image) - using BGR 
        t0 = time.time()
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            score.astype(np.int)
            box = box.astype(np.int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=1)
            # face = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]][:, :, ::-1]  # face position
            dRect = dlib.rectangle(left=box[0], top=box[1],
                                right=box[2], bottom=box[3])  # transform Opencv rectangle to dlib rectangle format
            shape = self.sp(frame, dRect)  #get landmarks
            face_desc0 = self.model.compute_face_descriptor(frame, shape, 1)  # compute face descriptor
            distance = []
            for face_desc in self.FACE_DESC:
                distance.append(np.linalg.norm(np.array(face_desc) - np.array(face_desc0)))   # calculate distance between facebank and prediction
            distance = np.array(distance)
            idx = np.argmin(distance)
            if distance[idx] < 0.4:
                name = self.FACE_NAME[idx]
                cv2.putText(frame, name, (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7,(255, 255, 255), 2)
            else:
                cv2.putText(frame, 'unknow', (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7,(255, 255, 255), 2)
        except:
            pass
        # cv2.imshow("", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     print("Terminate by user")
        #     break
        t1 = time.time()
        print("frame")
        print(f'took {round(t1 - t0, 3)} to process')
