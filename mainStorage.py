# export GOOGLE_APPLICATION_CREDENTIALS="/Users/admin/Google Drive/Colab Notebooks/head-pose-module/deepheadposeapp-7b346656fe5c.json"

from CloudStorage.services import download_blob, upload_blob

if __name__ == '__main__':
    bucket_name = 'deepheadposeapp.appspot.com'
    download_blob(bucket_name,'videos/user_id/process_id/results/classroom.mp4','Test/newClassroom.mp4')
    # upload_blob(bucket_name,'Test/classroom.mp4','videos/user_id/process_id/results/classroom.mp4')
