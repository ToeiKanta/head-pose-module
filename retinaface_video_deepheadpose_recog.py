import json
import sys

import cv2
import numpy as np
import argparse
import os.path as osp
import os
from face_detection import RetinaFace
import headpose_module
from face_recognition import FaceRecognition_module
from timer import Timer
from face_recognition import Recog_feature_extract_module
from bird_eye_view import BirdEyeModuleOnlyHead
from deep_head_pose import DeepHeadposeModule
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

def get_project_dir():
    current_path = os.path.abspath(os.path.join(__file__, "../"))
    return current_path

def relative(path):
    path = os.path.join(get_project_dir(), path)
    return os.path.abspath(path)

class JsonSave():
    ### data = [x, y, z, roll, pitch, yaw]
    ### user = username
    ### pose = array(user, data)
    ### frame = frameNumber
    ### Items = array(frame, pose)
    jsonData = {
        'Items': [
            # {
            #     'frame': 0,
            #     'pose': [
            #         {
            #             'user': 'username',
            #             'data': [0,0,0,0,0,0], # x,y,z,yaw,pitch,roll
            #         }
            #     ]
            # }
        ]
    }

    pose = []

    def saveFrame(self,frameNumber):
        self.jsonData['Items'].append({
            'frame': frameNumber,
            'pose': self.pose
        })
        self.pose = []

    def addDataToFrame(self,username,x,y,z,yaw,pitch,roll):
        self.pose.append({
            'user': username,
            'data': [x, y, z, yaw, pitch, roll]
        })
class MyFunc():
    def deletedAllInputOutputFile(self,input,output):
        print("\nremove all files in: " + input + " out:" + output)
        os.remove(input)
        os.remove(output);
        os.remove(output + ".json");
        os.remove(output + "-unity.json");
# moving average history
class History():
    history = {'username':{'lm': [], 'bbox': [], 'rvec': [], 'tvec': [], 'cm': [], 'dc': []}}
    
    def get_history(self,username):
        return self.history[username]

    def set_history(self,username, history):
        self.history[username] = history

    def create_history(self,username):
        self.history[username] = {'lm': [], 'bbox': [], 'rvec': [], 'tvec': [], 'cm': [], 'dc': []}
    
    def remove_history(self,username):
        self.history.pop(username,None)
        # print(f"history len: {len(self.history)}")
class Color():
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)

if __name__ == "__main__":
    t = Timer()
    # for saving position
    boxId = 0
    users_in_img = [] # [[name,centerpoint(x,y),box,step]]
    # head-pose
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', metavar='FILE', dest='input_file', default='./Test/Class.mp4', help='Input video.')
    parser.add_argument('-o', metavar='FILE', dest='output_file', default='./output/Test', help='Output video.')
    parser.add_argument('-fd','--force-delete', dest='force_delete',action='store_true',help='Force delete video output if existed.')
    parser.add_argument('-wh', metavar='N', dest='wh', default=[720, 480], nargs=2, help='Frame size.')
    parser.add_argument('-lt', metavar='N', dest='landmark_type', type=int, default=3, help='Landmark type.')
    parser.add_argument('-dlip','--use-dlip-lm',action='store_true', dest='use_dlip_lm', help='using dlip landmark.')
    parser.add_argument('-lp', metavar='FILE', dest='landmark_predictor', default='model/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
    parser.add_argument('-sf', metavar='N', dest='start_frame', type=int, default=0, help='Start frame number.')
    parser.add_argument('-y','--yaw', metavar='N', dest='yaw', type=int, default=0, help='set yaw.')
    parser.add_argument('-p','--pitch', metavar='N', dest='pitch', type=int, default=0, help='set pitch.')
    parser.add_argument('-r','--roll', metavar='N', dest='roll', type=int, default=0, help='set raw.')
    parser.add_argument('-fl', metavar='N', dest='frame_limit', type=int, default=-1, help='Frame limit. (Default - will play until ended)')
    parser.add_argument('-fs','--frame-skip', metavar='N', dest='frame_skip_number', type=int, default=1, help='Frame skip number. (Default - 1)')
    parser.add_argument('-flip', action='store_true', dest='flip_video', help='Flip video.')
    parser.add_argument('-scale', metavar='FLOAT', dest='video_scale', type=float, default=1.0, help='Video scale.')
    parser.add_argument('-nr','--no-recog',action='store_true', dest='close_recognition', help='Close Recognition mode.')
    parser.add_argument('-nt','--no-train',action='store_true', dest='close_recognition_training', help='Close Recognition Training mode.')
    parser.add_argument('-nb','--no-bird',action='store_true', dest='close_bird_eye', help='Close BirdEye Mode.')
    parser.add_argument('-ni','--no-img-show',action='store_true', dest='close_show_image', help='Close Image Realtime show.')
    parser.add_argument('-cpu','--use-cpu',action='store_true', dest='use_cpu', help='Use CPU.')
    parser.add_argument('-ph','--plane-height',dest='plane_height',type=int,default=0, help='Set your plane height from the floor.')
    parser.add_argument('-fb', '--use-firebase',action='store_true', dest='use_firebase',help='Set boolean to use firebase.')
    # args = vars(parser.parse_args())
    parser.add_argument('-uid', '--user-id', dest='uid', type=str, required=True,help='Set firestore user_id to save data.')
    parser.add_argument('-pid', '--process-id', dest='pid', type=str, required=True,
                        help='Set firestore process_id to save data.')
    args = vars(parser.parse_args())
    # hpd = headpose_module.HeadposeDetection(args["landmark_type"], args["landmark_predictor"])
    # close head-pose
    print('close_recognition : {}'.format(args["close_recognition"]))
    print('close_bird_eye : {}'.format(args["close_bird_eye"]))
    plane_height = args["plane_height"]
    closeBirdEye = args["close_bird_eye"]
    closeImShow = args["close_show_image"]
    frameSkipNumber = args["frame_skip_number"]
    isRecognition = not args["close_recognition"]
    filename = args["input_file"]
    use_dlip_lm = args["use_dlip_lm"]
    scale = args["video_scale"]
    isNoTrain = args["close_recognition_training"]
    outputPath = args["output_file"]
    useCPU = args["use_cpu"]
    useFirebase = args["use_firebase"]
    ## firebase
    # if True:
    try:
        if useFirebase:
            # Use a service account
            cred = credentials.Certificate(
                relative('firebase_cloud_functions/deepheadposeapp-firebase-adminsdk-ifvng-e13ae57e49.json'))
            firebase_admin.initialize_app(cred)

            db = firestore.client()
            firebase_uid = args["uid"]
            firebase_pid = args["pid"]
            process_ref = db.collection(u'processes').document(firebase_pid)
            process = process_ref.get()
            print(f'\nuid:{firebase_uid} pid:{firebase_pid}')
            if process.exists:
                processDict = process.to_dict()
                print(f'\nDocument data: {processDict}')
            else:
                print(f'\nNo such document!')
                MyFunc.deletedAllInputOutputFile(filename, outputPath);
                exit(1)
        ##
        if not useCPU:
            # Initialize head pose detection
            deepHeadPose = DeepHeadposeModule()
        historySave = History()
        if useCPU:
            detector = RetinaFace(gpu_id=-1)
        else:
            detector = RetinaFace(gpu_id=0)
        cap = cv2.VideoCapture(filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        name, ext = osp.splitext(filename)

        if args["force_delete"] and osp.exists(outputPath):
            os.remove(outputPath)
        elif osp.exists(outputPath):
            ans = ''
            while ans != 'y' and ans != 'n' :
                ans = input("File output exist, do you want to replace? (y/n) : ")
            if ans == 'y':
                os.remove(outputPath)
            else:
                exit()
        if not closeBirdEye:
            v_w = int(width * scale)+2*int(height * scale/2)
            v_h = int(height * scale)
            bird_w = 800
            birdEye = BirdEyeModuleOnlyHead(output_dir=os.path.abspath('./out'),output_vid=os.path.abspath(outputPath),video_path=os.path.abspath(filename),scale = scale,opencv = cv2, closeImShow = closeImShow, bird_width = bird_w, bird_height = v_h, plane_height = plane_height)
            ## bird_img = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
            out = cv2.VideoWriter(outputPath, fourcc, fps, (v_w + bird_w, v_h))
        else:
            out = cv2.VideoWriter(outputPath, fourcc, fps, (int(width*scale), int(height*scale)))
        # high performance at w: 540.0 h: 960.0
        print(f'Main: origin w: {width*scale} h: {height*scale}')

        # intitial frame
        jsonSave = JsonSave()
        start_frame = args["start_frame"]
        count = start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        frame_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        saveEyePoints = []
        while (cap.isOpened()):
            t.tic('FF')
            ret, img = cap.read()
            if(count % frameSkipNumber != 0): # process every n frame
                count += 1
                continue
            if args['flip_video']:
                img = cv2.flip(img,0)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is None:
                break
            img = cv2.resize(img, (int(width*scale) , int(height*scale)))
            # expand image horizontal for easy to setup bird view space
            if not closeBirdEye:
                h, w = img.shape[:2]
                if(useFirebase):
                    img = cv2.copyMakeBorder(img, int(h * processDict["setup"]["padding_y"]), int(h * processDict["setup"]["padding_y"]), int(w * processDict["setup"]["padding_x"]), int(w * processDict["setup"]["padding_x"]), cv2.BORDER_CONSTANT,
                                             value=[255, 255, 255])  # top, bottom, left, right
                    if count == start_frame: # run setup only firstFrame
                        h, w = img.shape[:2]
                        birdEye.setupFirebase(processDict, w, h)
                else:
                    img = cv2.copyMakeBorder(img, 0, 0, int(h/2), int(h/2), cv2.BORDER_CONSTANT, value=[255, 255, 255]) # top, bottom, left, right
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # <note> img หลังจากนี้ จะมี padding แล้ว
            h, w = img.shape[:2]
            if count == start_frame:  # run setup only firstFrame
                print(f'\nMain: img + padding w: {w} h: {h}')
                print('\nMain: Calcurate YAW added : ' + str(birdEye.getYawAdded()))
            faces = detector(img_rgb)
            # print(f'find face : {len(faces)}\n')
            used_face = 0
            boxs = []
            user_names = []
            rotations = []
            direction_points = []

            for box, landmarks, score in faces: # box = x,y,w,h โดย frame[y:h, x:w]
                if score <= 0.3:
                    # print(f'\n skipped score <= 0.2 \n')
                    continue
                boxs.append(box)
                box = box.astype(np.int)
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                # x2 = x-int((w-x)/5)
                # if x2 <= 0:
                #     x2 = 0
                # y2 = y-int((h-y)/5)
                # if y2 <= 0:
                #     y2 = 0
                # w2 = w + int((w-x)/5)
                # h2 = h + int((h-y)/5)

                # box[0] = x2
                # box[1] = y2
                # box[2] = w2
                # box[3] = h2

                cropped = img[y:h,x:w]
    ######### For Recognition #########
                t.tic('REC')
                center_point = ((w+x)/2,(h+y)/2)
                isSamePos = False
                user_name = ''
                user_index = 0
                for users in users_in_img:
                    nname,bbox,ccenter,step = users[:4]
                    if (center_point[0]>=bbox[0] and center_point[0]<=bbox[2]) and (center_point[1]>=bbox[1] and center_point[1]<=bbox[3]):#(count-start_frame)%1 == 0 :
                        # use your saving user
                        users_in_img[user_index] = [users[0],users[1],users[2],0] # reset step
                        user_name = nname # load save
                        isSamePos = True
                        break
                    user_index += 1
                if isSamePos:
                    if user_name == 'unknow':
                        # if user not founded -> find who is him?
                        if isRecognition and (count-start_frame)%10 == 0:
                            face_recognition = FaceRecognition_module()
                            user_name = face_recognition.detect(frame=img,box=box,landmarks=landmarks, score=score)
                        if not isNoTrain:
                            if not closeImShow:
                                cv2.imshow('img', cropped)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                            n = input("Please enter name: ")
                            if n != 'pass!':
                                feature_extract = Recog_feature_extract_module()
                                feature_extract.train(n,cropped)
                                user_name = n
                    users_in_img[user_index] = [user_name,box,center_point,0] # update position data
                else:
                    # if new pos -> find who is him?
                    boxId += 1
                    user_name = "b:"+str(boxId)
                    users_in_img.append([user_name,box,center_point,0]) # save new human position
                    historySave.create_history(user_name) # create new head-pose history
                    if isRecognition and (count-start_frame)%10 == 0:
                        face_recognition = FaceRecognition_module()
                        user_name = face_recognition.detect(frame=img,box=box,landmarks=landmarks, score=score)
                        users_in_img.append([user_name,box,center_point,0])
    ######### Close Recognition #########

                # print(f'\nuser: {user_name}')
                # print('REC: %.2f' % t.toc('REC'), end='ms')
                # Display the resulting frame
                yaw,pitch,roll = (0,0,0) ## for test on cpu, we will mockup data rotation
                if not useCPU:
                    yaw,pitch,roll,new_img = deepHeadPose.getPose(frame=img,box=box)
                    img = new_img
                    if not closeBirdEye:
                        rotations.append((yaw + birdEye.getYawAdded() ,pitch + args["pitch"],roll + args["roll"]));
                else:
                    #### if using CPU
                    rotations.append((12.3872, -108.2779, 6.8547));
                # img, angles, new_history = hpd.process_image(img,box,True,1)
                user_names.append(user_name);
                # width, height = cropped.shape[:2]
                # print(f'w: {width} h: {height}')

                if img is None:
                    break
                else:
                    ### draw retina facial landmarks #######
                    for p in landmarks:
                        point = tuple(p.astype(int))
                        # cv2.circle(img, point, 1, Color.yellow,-1)
                # draw head detector
                cv2.rectangle(
                    img, (x,y), (w,h), color=(0, 255, 0), thickness=2
                )
                used_face += 1
    ######### Draw Position Saved #########
            i = 0
            user_list = ''
            for users in users_in_img:
                nname,bbox,ccenter,step = users[:4]
                if step >= 3:
                    # print(f'\npop => {users_in_img.pop(i)} name => {nname}')
                    historySave.remove_history(nname)
                    continue
                user_list += users[0] + ' '
                users_in_img[i] = [users[0],users[1],users[2],users[3]+1] # step up
                i += 1 # count index
                # Write Box's name from saving position
                # if step == 0:
                    # cv2.putText(img, users[0] + "(" + str(users[3]) + ")", (int(users[2][0]), int(users[2][1]) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.2,(0, 255, 0), 1)
            # print(f' boxs: {user_list} ')
    ######### Close Draw Position Saved #########

    ######### Show Bird Eye View #########
            if not closeBirdEye:
                ## birdEyeImg = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
                birdEyeImg, eyePoints, personPoints = birdEye.calculate_social_distancing_retina_box(boxs, img, rotations)
                saveEyePoints.append(eyePoints)

                # pad = np.full((img.shape[0],700,3), [255, 255, 255], dtype=np.uint8)
                #cv2.putText(pad, "-- HIGH RISK : " + str(risk_count[0]) + " people", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                #cv2.putText(pad, "-- LOW RISK : " + str(risk_count[1]) + " people", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                #cv2.putText(pad, "-- SAFE : " + str(risk_count[2]) + " people", (50,  80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # print(f"\nMain: BirdEyeImg ({birdEyeImg.shape[1]},{birdEyeImg.shape[0]}) img ({img.shape[1]},{img.shape[0]})")
                img = np.hstack((birdEyeImg, img))
                # cv2.imshow('img', img)
    ######### Close Show Bird Eye View #########
    ######### Json Saving ###########
                i = 0
                for rov in rotations:
                    jsonSave.addDataToFrame(user_names[i], personPoints[i][0], personPoints[i][1], plane_height,  float(rov[0]), float(rov[1]), float(rov[2]));
                    i += 1
            ######### save firebase status process ###########
            if useFirebase and count % 10 == 0:
                processDict['status'] = u"PROCESS"
                processDict['error_msg'] = u""
                processDict['percent'] = count/frame_len*100
                process_ref.update(processDict)
            ######### close save status process ###########
            # print(f'\rused face : {used_face}\n')
            # print('\rframe: %d \n' % count, end='')
            print(f' frame: {count} count: {count-start_frame}')
            print(' time: %.2f ' % t.toc('FF'), end='ms')
            if not closeImShow:
                cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                headpose_module.t.summary()
                break
            jsonSave.saveFrame(count);
            out.write(img)#cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            count += 1
            # close head-pose
            if args["frame_limit"] != -1 and count - start_frame >= args["frame_limit"]:
                break

        with open(outputPath + ".json", "w") as savefile:
            json.dump(saveEyePoints, savefile)
        with open(outputPath + "-unity.json", "w") as savefile:
            json.dump(jsonSave.jsonData, savefile)
        ######### save firebase status process ###########
        if useFirebase:
            processDict['status'] = u"SUCCESS"
            processDict['percent'] = 100
            process_ref.update(processDict)
            MyFunc.deletedAllInputOutputFile(filename,outputPath);
        ######### close save status process ##########
        # When everything done, release the capture
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f'\nEnd Success:')
    except:
        print(f'\nFailed with {str(sys.exc_info()[0])}')
        if useFirebase:
            processDict['status'] = u"FAILED"
            processDict['percent'] = 100
            processDict['error_msg'] = str(sys.exc_info()[0])
            MyFunc.deletedAllInputOutputFile(filename,outputPath);
            process_ref.update(processDict)

