import json
import boto3
import logging
from typing import Dict, Any
import datetime


# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:

    s3_client = boto3.client('s3')
    sagemaker_client = boto3.client('sagemaker')

    # Define the S3 bucket and config file key
    source_bucket = event.get('source_bucket', 'qloudy-xgboost-demo')
    source_config_key = event.get('source_config_key')
    debug_ = event.get('debug_', False)

    # Fetch the processing config file from S3
    try:
        config_object = s3_client.get_object(Bucket=source_bucket, Key=source_config_key)
        config_data = json.loads(config_object['Body'].read().decode('utf-8'))
    except Exception as e:
        logger.error(f"Error fetching or parsing the configuration file from S3: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps('Error fetching or parsing the configuration file from S3.')
        }

    try:
        requirements_uri = config_data['requirements']['s3_uri']
        processing_script_s3_uri = config_data['processing_script']['s3_uri']
        input_data_location = config_data['input_data_location']['s3_uri']
        output_hpo_location = config_data['output_hpo_location']['s3_uri']
        output_training_location = config_data['output_training_location']['s3_uri']
        wine_quality_metadata_location = config_data['wine_quality_metadata_location']['s3_uri']
        InstanceType = config_data['InstanceType']
        ImageUri = config_data['ImageUri']
        ContainerEntrypointScript = config_data['ContainerEntrypointScript']
        role = config_data['role']

    except KeyError as e:
        missing_field = e.args[0]
        logger.error(f"Missing required field in the configuration file: {missing_field}")
        return {
            'statusCode': 400,
            'body': json.dumps(f'Missing required field in the configuration file: {missing_field}')
        }

    # Generate a dynamic processor job name
    offset = datetime.timedelta(hours=2)
    utc_now = datetime.datetime.utcnow()
    madrid_now = utc_now + offset
    date = madrid_now.strftime('%m-%d-%H%M%S')
    processor_job_name = "wine-quality-processor-" + date

    # Define processing job arguments
    processing_args = {
        'ProcessingInputs': [
            {
                'InputName': 'requirements-input',
                'S3Input': {
                    'S3Uri': requirements_uri,
                    'LocalPath': '/opt/ml/processing/input/requirements',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                    'S3DataDistributionType': 'FullyReplicated'
                }
            },
            {
                'InputName': 'code-input',
                'S3Input': {
                    'S3Uri': processing_script_s3_uri,
                    'LocalPath': '/opt/ml/processing/input/code',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                    'S3DataDistributionType': 'FullyReplicated'
                }
            },
            {
                'InputName': 'data-input',
                'S3Input': {
                    'S3Uri': input_data_location,
                    'LocalPath': '/opt/ml/processing/input/dataset',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                    'S3DataDistributionType': 'FullyReplicated'
                }
            }
        ],
        'ProcessingOutputConfig': {
            'Outputs': [
                {
                    'OutputName': 'preprocessed_hpo_data',
                    'S3Output': {
                        'S3Uri': output_hpo_location,
                        'LocalPath': '/opt/ml/processing/output/hpo',
                        'S3UploadMode': 'EndOfJob'
                    }
                },
                {
                    'OutputName': 'preprocessed_training_data',
                    'S3Output': {
                        'S3Uri': output_training_location,
                        'LocalPath': '/opt/ml/processing/output/training',
                        'S3UploadMode': 'EndOfJob'
                    }
                },
                {
                    'OutputName': 'wine_quality_metadata',
                    'S3Output': {
                        'S3Uri': wine_quality_metadata_location,
                        'LocalPath': '/opt/ml/processing/output/wine_quality_df_metadata',
                        'S3UploadMode': 'EndOfJob'
                    }
                }
            ]
        },
        'ProcessingResources': {
            'ClusterConfig': {
                'InstanceCount': 1,
                'InstanceType': InstanceType,
                'VolumeSizeInGB': 30
            }
        },
        'AppSpecification': {
            'ImageUri': ImageUri,
            'ContainerEntrypoint': ['python3', ContainerEntrypointScript,'-d', input_data_location]
        },
        'RoleArn': role,
        'ProcessingJobName': processor_job_name,
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 3600
        }
    }

    # Create the processing job if debug mode is disabled
    if not debug_:
        try:
            response = sagemaker_client.create_processing_job(**processing_args)
        except Exception as e:
            logger.error(f"Error creating processing job: {e}")
            return {
                'statusCode': 500,
                'body': json.dumps('Error creating processing job.')
            }
    else:
        logger.info('Debug mode ')
        processor_job_name = ''
        response = None


    
    results={
        "ProcessingJobName":processor_job_name,
        "source_bucket":source_bucket
    }
    
    return results