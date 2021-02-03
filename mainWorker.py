# export GOOGLE_APPLICATION_CREDENTIALS="/Users/admin/Google Drive/Colab Notebooks/head-pose-module/deepheadposeapp-7b346656fe5c.json"
## for test
# videos/user_id/process_id/results/classroom.mp4
import logging
from CloudPubSub.consts import consts
from CloudPubSub.subscriber import listen
from CloudStorage.services import download_blob, upload_blob
import time
import os
import subprocess

def record_job_status(message, status, err_msg, return_vals):
    # return_vals is for communication with worker
    return_vals['status'] = status
    return_vals['err_msg'] = err_msg

    # may also consider recording the job status in your mysql database
    # update_job_status(message, status)  # not implement

def test_step(msg):
    logging.info("[Test run] message : %s\n", msg)
    t_end = time.time() + 20
    while time.time() < t_end:
        # do whatever you do
        pass
    status = consts.DONE_STATUS
    err_msg = ""
    return status, err_msg

def download_video(video_path):
    logging.info("[run step 1] Download video from : %s\n", video_path)
    videoName = os.path.basename(video_path)
    logging.info(video_path)
    logging.info(videoName)
    download_blob('default', video_path, 'Test/' + videoName)
    status = consts.DONE_STATUS
    err_msg = ""
    return status, err_msg

def process_video(video_path, start_frame = 3000,frame_length = 550,scale = 0.8):
    logging.info("[run step 2] Process_video : %s\n", video_path)
    subprocess.check_call(['python', 'retinaface_video_deepheadpose_recog.py', '-i', 'Test/'+os.path.basename(video_path), '-o' ,'output/' + os.path.basename(video_path), '-ni' ,'-cpu', '-fl', str(frame_length), '-nr' ,'-sf', str(start_frame),'-nt', '-scale',str(scale), '-fd' ,'-p', '-80','-ph','80'],shell=False)
    status = consts.DONE_STATUS
    err_msg = ""
    return status, err_msg

def upload_video(src_path, video_path):
    logging.info("[run step 3] Upload video from : %s to : %s\n", video_path, src_path)
    upload_blob('default', src_path, video_path)
    status = consts.DONE_STATUS
    err_msg = ""
    return status, err_msg

@listen
def start_service(return_vals, message):
    try:
        # logging.info("Processing message: %s" % message.message.data)
        video_path = message.message.data.decode("utf-8")

        status, err_msg = download_video(video_path)
        if status!= consts.DONE_STATUS:
            logging.error("Failed to run step1 - Download video!")
            record_job_status(message, status, err_msg, return_vals)
            return
        status, err_msg = process_video(video_path)
        if status != consts.DONE_STATUS:
            logging.error("Failed to run step2! - Process video")
            record_job_status(message, status, err_msg, return_vals)
            return

        src_path = 'output/' + os.path.basename(video_path)
        video_upload_path = 'videos/user_id/process_id/results/' + os.path.basename(video_path) + '-result.mov'
        status, err_msg = upload_video(src_path, video_upload_path)
        if status != consts.DONE_STATUS:
            logging.error("Failed to run step3! - Upload video result")
            record_job_status(message, status, err_msg, return_vals)
            return
        logging.info("Message completely processed: %s" % message.message.data)
        # may also consider recording the job status in your mysql database
        # update_job_status(message, "DONE")  # not implement

        return_vals['status'] = consts.DONE_STATUS
        return_vals['err_msg'] = ""
    except:
        logging.info("Process message failed: \n%s" % message.message.data)
        return_vals['status'] = consts.ERROR_UNKNOWN
        return_vals['err_msg'] = "unknown"

        # may also consider recording the job status in your mysql database
        # update_job_status(message, "FAILED")  # not implement
        raise

if __name__=='__main__':
    try:
        # create console handler and set level to debug
        data_logger = logging.getLogger()
        data_logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        data_logger.addHandler(ch)

        start_service()
    except:
        logging.info("failed to start service...")
        raise