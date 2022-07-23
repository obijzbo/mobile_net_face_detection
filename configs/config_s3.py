import json
from configs.config_ml_model import ROOT_DIR

with open(f'{ROOT_DIR}/credentials.json') as credentials_file:
    credentials = json.load(credentials_file)

AWS_ACCESS_KEY_ID = credentials['s3_ml_model']['access_key_id']
AWS_SECRET_ACCESS_KEY = credentials['s3_ml_model']['secret_access_key']
AWS_STORAGE_BUCKET_NAME = credentials['s3_ml_model']['bucket_name']
DEFAULT_FILE_STORAGE = credentials['s3_ml_model']['default_file_storage']
AWS_S3_REGION_NAME = credentials['s3_ml_model']['s3_rigion_name']