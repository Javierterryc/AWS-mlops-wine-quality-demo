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
    eval_metric = event.get('eval_metric')
    model_package_group_name = event.get('model_package_group_name')
    
    training_metadata_name = f'training_metadata-{date}.json'
    s3_key = f'wine-quality-project/pipeline-metadata/training-job-metadata/{training_metadata_name}'
    
    try:
        training_job_name = event["TrainingJobName"]
    except KeyError as e:
        logger.error(e)
        logger.error("TrainingJobName not found in the event payload.")
        return {
            'statusCode': 400,
            'body': json.dumps('TrainingJobName not found in the event payload.')
        }

    try:
        response = sm_client.describe_training_job(TrainingJobName=training_job_name)

        training_start_time = response['TrainingStartTime']
        training_end_time = response['TrainingEndTime']
        training_duration = (training_end_time - training_start_time).total_seconds()

        # Define metadata file structure
        training_metadata = {
            'ModelRegistry':{
                'ModelDataUrl': response['ModelArtifacts']['S3ModelArtifacts'],
                'Hyperparameters': response['HyperParameters'],
                'ModelMetrics': response['FinalMetricDataList']
                
            },
            'TrainingJobMetadata':{
                'TrainingJobName': response['TrainingJobName'],
                'TrainingJobArn': response['TrainingJobArn'],
                'TrainingJobStatus': response['TrainingJobStatus'],
                'TrainingStartTime': str(training_start_time),
                'TrainingEndTime': str(training_end_time),
                'TrainingDurationInSeconds': training_duration,
                'TrainingImage': response['AlgorithmSpecification']['TrainingImage']
            }
        }

        json_data = json.dumps(training_metadata, default=str, indent=2)

        # Upload metadata file to S3 path
        s3_client.put_object(
            Bucket=source_bucket,
            Key=s3_key,
            Body=json_data.encode('utf-8')
        )
        logger.info(f"Successfully uploaded training metadata to S3 bucket {source_bucket} with key {s3_key}.")
        
        return {
            'statusCode': 200,
            'body': json.dumps('Training metadata uploaded successfully.'),
            'TrainingMetadataJson': training_metadata_name,
            'source_bucket': source_bucket,
            'eval_metric': eval_metric,
            'model_package_group_name':model_package_group_name
        }
    
    except ClientError as e:
        logger.error(f"ClientError: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Failed to retrieve or upload training metadata: {e}")
        }
    
    except Exception as e:
        logger.error(f"Exception: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"An error occurred: {e}")
        }