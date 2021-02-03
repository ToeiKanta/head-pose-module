import logging
from CloudPubSub.consts import consts
from CloudPubSub.subscriber import listen
import time

def record_job_status(message, status, err_msg, return_vals):
    # return_vals is for communication with worker
    return_vals['status'] = status
    return_vals['err_msg'] = err_msg

    # may also consider recording the job status in your mysql database
    # update_job_status(message, status)  # not implement

def run_step1(msg):
    logging.info("[run step 1] message : %s\n", msg)
    t_end = time.time() + 20
    while time.time() < t_end:
        # do whatever you do
        pass
    status = consts.DONE_STATUS
    err_msg = ""
    return status, err_msg

@listen
def start_service(return_vals, message):
    try:
        logging.info("Processing message: %s" % message.message.data)

        status, err_msg = run_step1(message.message.data)
        if status!= consts.DONE_STATUS:
            logging.error("Failed to run step1!")
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