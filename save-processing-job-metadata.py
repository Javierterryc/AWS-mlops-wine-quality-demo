import json
import logging
import boto3
from botocore.exceptions import ClientError
import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Set datetime
offset = datetime.timedelta(hours=2)
utc_now = datetime.datetime.utcnow()
madrid_now = utc_now + offset
date = madrid_now.strftime('%m-%d-%H%M%S')


def get_last_modified(df_prefix,source_bucket,s3_client):
    '''
    Retreive the last added/modified object in an S3Uri
    '''
    response = s3_client.list_objects_v2(Bucket=source_bucket, Prefix=df_prefix)
    contents = response['Contents']
    contents_sorted = sorted(contents, key=lambda obj: obj['LastModified'])
    last_df_key = contents_sorted[-1]['Key']
    last_df_S3uri = f"s3://{source_bucket}/{last_df_key}"
    return last_df_S3uri

def lambda_handler(event, context):
    
    s3_client = boto3.client('s3')
    sm_client = boto3.client('sagemaker')
    source_bucket = event.get('source_bucket')
    s3_key = f'wine-quality-project/pipeline-metadata/processing-job-metadata/processing_metadata-{date}.json'
    processing_job_name = event["ProcessingJobName"]
    
    if not processing_job_name:
        logger.error("ProcessingJobName not found in the event payload.")
        return {
            'statusCode': 400,
            'body': json.dumps('ProcessingJobName not found in the event payload.')
        }

    try:
        response = sm_client.describe_processing_job(ProcessingJobName=processing_job_name)

        processing_start_time = response['ProcessingStartTime']
        processing_end_time = response['ProcessingEndTime']
        processing_duration = (processing_end_time - processing_start_time).total_seconds()
        
        # Get last training dfs
        last_train_df_prefix = 'wine-quality-project/preprocessed_data/training/train'
        last_train_df_S3Uri = get_last_modified(last_train_df_prefix,source_bucket,s3_client)
        
        last_test_df_prefix = 'wine-quality-project/preprocessed_data/training/test'
        last_test_df_S3Uri = get_last_modified(last_test_df_prefix,source_bucket,s3_client)
        
        # Get last HPO dfs
        last_hpo_train_df_prefix = 'wine-quality-project/preprocessed_data/hpo/train'
        last_hpo_train_df_S3Uri =  get_last_modified(last_hpo_train_df_prefix,source_bucket,s3_client)
        
        last_hpo_test_df_prefix = 'wine-quality-project/preprocessed_data/hpo/test'
        last_hpo_test_df_S3Uri = get_last_modified(last_hpo_test_df_prefix,source_bucket,s3_client)
        
        date_id = '-'.join(last_train_df_S3Uri.split('-')[-3:]).split('.')[0]

        processing_metadata = {
            'ProcessingJobName': response['ProcessingJobName'],
            'ProcessingJobArn': response['ProcessingJobArn'],
            'ProcessingJobStatus': response['ProcessingJobStatus'],
            'ProcessingStartTime': str(processing_start_time),
            'ProcessingEndTime': str(processing_end_time),
            'ProcessingDurationInSeconds': processing_duration,
            'DatasetProperties' : {
                'DatasetName': f'wine_quality-{date_id}',
                's3_prefix': 'wine-quality-project/preprocessed_data/training',
                'train_uri': last_train_df_S3Uri,
                'test_uri': last_test_df_S3Uri,
                'hpo_train_uri':last_hpo_train_df_S3Uri,
                'hpo_test_uri':last_hpo_test_df_S3Uri
            }
        }

        json_data = json.dumps(processing_metadata, default=str, indent=2)

        s3_client.put_object(
            Bucket=source_bucket,
            Key=s3_key,
            Body=json_data.encode('utf-8')
        )
        
        logger.info(f"Successfully uploaded processing metadata to S3 bucket '{source_bucket}' with key'{s3_key}'.")
        
        return {
            'statusCode': 200,
            'body': json.dumps('Processing metadata uploaded successfully.'),
            'source_bucket':source_bucket
        }
    
    except ClientError as e:
        logger.error(f"ClientError: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Failed to retrieve or upload processing metadata: {e}")
        }
    
    except Exception as e:
        logger.error(f"Exception: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"An error occurred: {e}")
        }