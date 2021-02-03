from google.cloud import storage # pip install --upgrade google-cloud-storage
# export GOOGLE_APPLICATION_CREDENTIALS="/Users/admin/Google Drive/Colab Notebooks/head-pose-module/deepheadposeapp-7b346656fe5c.json"

def download_blob(bucket_name, source_blob_name, destination_file_name):
    if bucket_name == 'default':
        bucket_name = 'deepheadposeapp.appspot.com'
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

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

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    if bucket_name == 'default':
        bucket_name = 'deepheadposeapp.appspot.com'
    """Uploads a file to the bucket."""
    # bucket_name = "deepheadposeapp.appspot.com"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )
