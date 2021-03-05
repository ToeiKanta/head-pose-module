# from google.cloud import storage # pip install --upgrade google-cloud-storage
# import firebase_admin
# from firebase_admin import credentials
from google.cloud import storage
# import os
# # def get_project_dir():
# #     current_path = os.path.abspath(os.path.join(__file__, "../"))
# #     return current_path
# #
# # def relative(path):
# #     path = os.path.join(get_project_dir(), path)
# #     return os.path.abspath(path)
# #
# # cred = credentials.Certificate(relative("../deepheadposeapp-7b346656fe5c.json"))
# # firebase_admin.initialize_app(cred, {
# #     'storageBucket': 'deepheadposeapp.appspot.com'
# # })
# export GOOGLE_APPLICATION_CREDENTIALS="/Users/admin/Google Drive/Colab Notebooks/head-pose-module/deepheadposeapp-7b346656fe5c.json"

def download_blob(bucket_name = 'deepheadposeapp.appspot.com', source_blob_name = "", destination_file_name = ""):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    print("\nCloudStorage: Download file from: " + source_blob_name + " to:" + destination_file_name)
    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )

def upload_blob(bucket_name = 'deepheadposeapp.appspot.com', source_file_name = "", destination_blob_name = ""):
    """Uploads a file to the bucket."""
    # bucket_name = "deepheadposeapp.appspot.com"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"
    print("\nCloudStorage: Upload file from: " + source_file_name + " to:" + destination_blob_name)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )
