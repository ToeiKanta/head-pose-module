
import cv2
import numpy as np

from face_detection import RetinaFace


if __name__ == "__main__":
    scale = 0.5
    detector = RetinaFace(gpu_id=-1)
    cap = cv2.VideoCapture('./Test/ZoomClass2.mp4')
    ret, img = cap.read(0)
    width, height = img.shape[:2]
    # high performance at w: 540.0 h: 960.0
    print(f'w: {width*scale} h: {height*scale}')
    while True:
        ret, img = cap.read(0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (int(height*scale) , int(width*scale)))
        faces = detector(img)
        for box, landmarks, score in faces:
            box = box.astype(np.int)
            cv2.rectangle(
                img, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=2
            )
        cv2.imshow("", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

