
import cv2
import numpy as np
import headpose
import argparse
import os.path as osp

from face_detection import RetinaFace


if __name__ == "__main__":
    # head-pose
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', metavar='FILE', dest='input_file', default=None, help='Input video. If not given, web camera will be used.')
    parser.add_argument('-o', metavar='FILE', dest='output_file', default=None, help='Output video.')
    parser.add_argument('-wh', metavar='N', dest='wh', default=[720, 480], nargs=2, help='Frame size.')
    parser.add_argument('-lt', metavar='N', dest='landmark_type', type=int, default=1, help='Landmark type.')
    parser.add_argument('-lp', metavar='FILE', dest='landmark_predictor', 
                        default='model/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
    args = vars(parser.parse_args())
    # Initialize head pose detection
    hpd = headpose.HeadposeDetection(args["landmark_type"], args["landmark_predictor"])
    # close head-pose 
    filename = './Test/Class.mp4'
    scale = 0.4
    detector = RetinaFace(gpu_id=-1)
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    name, ext = osp.splitext(filename)
    out = cv2.VideoWriter(args["output_file"], fourcc, fps, (int(width*scale), int(height*scale)))

    # high performance at w: 540.0 h: 960.0
    print(f'w: {width*scale} h: {height*scale}')

    count = 0
    while (cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img,0)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (int(width*scale) , int(height*scale)))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faces = detector(img_rgb)
        print(f'find face : {len(faces)}\n')
        for box, landmarks, score in faces: # box = x,y,w,h โดย frame[y:h, x:w]
            if score <= 0.2:
                # print(f'\n skipped score <= 0.2 \n')
                continue
            box = box.astype(np.int)
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            # head-pose
            x2 = x-int((w-x)/2)
            if x2 <= 0: 
                x2 = 0
            y2 = y-int((h-y)/2)
            if y2 <= 0: 
                y2 = 0
            w2 = w + int((w-x)/2)
            h2 = h + int((h-y)/2)
            cropped = img[y2:h2,x2:w2]
            # cv2.imshow('img', cropped)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     # headpose.t.summary()
            #     break
            # Display the resulting frame
            frame, angles = hpd.process_image(cropped)
            # width, height = cropped.shape[:2]
            # print(f'w: {width} h: {height}')
            
            if frame is None: 
                # draw head detector
                cv2.rectangle(
                    img, (x,y), (w,h), color=(255, 255, 255), thickness=1
                )
                break
            else:
                # frame = cv2.resize(cropped, (int(width*scale) , int(height*scale)))
                img[y2:h2,x2:w2] = frame;
            # draw head detector
            cv2.rectangle(
                img, (x,y), (w,h), color=(255, 0, 0), thickness=1
            )
            
        print('\rframe: %d' % count, end='')
        # cv2.imshow('img', img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     headpose.t.summary()
        #     break
        out.write(img)#cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        count += 1
        # close head-pose
        # if count >= 200:
        #     break
    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()