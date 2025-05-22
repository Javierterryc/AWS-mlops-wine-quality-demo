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

def lambda_handler(event, context):
    
    s3_client = boto3.client('s3')
    sm_client = boto3.client('sagemaker')
    source_bucket = event.get('source_bucket')
    
    batch_metadata_name = f'batch_metadata-{date}.json'
    s3_key = f'wine-quality-project/pipeline-metadata/batch-job-metadata/{batch_metadata_name}'
    
    try:
        batch_job_name = event.get('BatchJobName')
    except KeyError as e:
        logger.error(e)
        logger.error("BatchJobName not found in the event payload.")
        return {
            'statusCode': 400,
            'body': json.dumps('BatchJobName not found in the event payload.')
        }
    

    try:
        response_1 = sm_client.describe_transform_job(TransformJobName=batch_job_name)
        model_name = response_1['ModelName']
        response_2 = sm_client.describe_model(ModelName=model_name)

        transform_start_time = response_1['TransformStartTime']
        transform_end_time = response_1['TransformEndTime']
        transform_duration = (transform_end_time - transform_start_time).total_seconds()

        metadata = {
            'TransformJobMetadata': {
                'TransformJobName': response_1['TransformJobName'],
                'TransformJobArn': response_1['TransformJobArn'],
                'TransformJobDurationInMinutes':f'{round((transform_duration/60),2)} min',
                'TransformInputS3Uri': response_1['TransformInput']['DataSource']['S3DataSource']['S3Uri'],
                'TransformOutput': response_1['TransformOutput']
            },
            'ModelMetadata':{
                'ModelName': response_2['ModelName'],
                'Image': response_2['PrimaryContainer']['Image'],
                'ModelDataUrl': response_2['PrimaryContainer']['ModelDataUrl'],
                'ModelArn': response_2['ModelArn']
            }
        }

        json_data = json.dumps(metadata, default=str, indent=2)

        s3_client.put_object(
            Bucket=source_bucket,
            Key=s3_key,
            Body=json_data.encode('utf-8')
        )
        logger.info(f"Successfully uploaded batch metadata to S3 bucket {source_bucket} with key {s3_key}.")
        return {
            'statusCode': 200,
            'body': json.dumps('Batch metadata uploaded successfully.'),
            'TrainingMetadataJson': batch_metadata_name,
            'source_bucket': source_bucket
        }
    except ClientError as e:
        logger.error(f"ClientError: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Failed to retrieve or upload batch metadata: {e}")
        }
    except Exception as e:
        logger.error(f"Exception: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"An error occurred: {e}")
        }
