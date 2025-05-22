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

    try:
        tuning_job_name = event.get("HPOJobName")
    except KeyError as e:
        logger.error(e)
        logger.error("HyperParameterTuningJobName not found in the event payload.")
        return {
            'statusCode': 400,
            'body': json.dumps('HyperParameterTuningJobName not found in the event payload.')
        }
    
    best_hpo_metadata_name = f'best_hpo_job_metadata-{date}.json'
    s3_key = f'wine-quality-project/pipeline-metadata/hpo-job-metadata/{best_hpo_metadata_name}'
    
    try:
        response = sm_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)

        training_start_time = response['BestTrainingJob']['TrainingStartTime']
        training_end_time = response['BestTrainingJob']['TrainingEndTime']
        training_duration = (training_end_time - training_start_time).total_seconds()
        
        best_training_metadata = {
            'HyperParameterTuningJobMetadata':{
                "HyperParameterTuningJobName": response['HyperParameterTuningJobName'],
                "HyperParameterTuningJobArn":response['HyperParameterTuningJobArn'],
                "HyperParameterTuningJobObjective": response['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']
            },
            'TrainingJobMetadata':{
                'TrainingJobName': response['BestTrainingJob']['TrainingJobName'],
                'TrainingJobArn': response['BestTrainingJob']['TrainingJobArn'],
                'TrainingJobStatus': response['BestTrainingJob']['TrainingJobStatus'],
                'TrainingStartTime': str(training_start_time),
                'TrainingEndTime': str(training_end_time),
                'TrainingDurationInSeconds': training_duration,
                'TunedHyperParameters': response['BestTrainingJob']['TunedHyperParameters']
            }
        }

        json_data = json.dumps(best_training_metadata, default=str, indent=2)

        s3_client.put_object(
            Bucket=source_bucket,
            Key=s3_key,
            Body=json_data.encode('utf-8')
        )
        
        logger.info(f"Successfully uploaded hpo metadata to S3 bucket '{source_bucket}'' with key: '{s3_key}'.")
        return {
            'statusCode': 200,
            'body': json.dumps('Hpo metadata uploaded successfully.'),
            'source_bucket': source_bucket,
            'MetadataJsonName': best_hpo_metadata_name,
            'MetadataKey': 'wine-quality-project/pipeline-metadata/hpo-job-metadata/'
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
