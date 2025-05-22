import json
import boto3
import logging
from typing import Dict, Any
import datetime 
import os

# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_config_data(s3_client, bucket: str, key: str) -> Dict[str, Any]:
    '''
    This function downloads a file from the specified S3 bucket and key and then parses 
    it into a Python dictionary.
    
    Retreives batch config file
    '''
    try:
        config_object = s3_client.get_object(Bucket=bucket, Key=key)
        config_data = json.loads(config_object['Body'].read().decode('utf-8'))
        return config_data
    except s3_client.exceptions.NoSuchKey:
        logger.error(f"Configuration file not found: {key}")
        raise
    except Exception as e:
        logger.error(f"Error fetching or parsing the configuration file from S3: {e}")
        raise

def get_last_modified(prefix, bucket, s3_client):
    '''
    This function retreives the last modified object (AKA: the last added object) from an S3Uri.
    
    Retreives last batch df with no target for the batch job
    '''
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        contents = response['Contents']
        contents_sorted = sorted(contents, key=lambda obj: obj['LastModified'])
        last_object_key = contents_sorted[-1]['Key']
        return last_object_key
    except s3_client.exceptions.NoSuchKey:
        logger.error(f"Processing metadata file not found with prefix: {prefix}")
        raise
    except Exception as e:
        logger.error(f"Error fetching processing metadata from S3: {e}")
        raise

def create_batch_job(sagemaker_client, batch_job_args: Dict[str, Any]) -> Dict[str, Any]:
    '''
    Creates a batch transform job given the sagemaker client and the job args
    '''
    try:
        response = sagemaker_client.create_transform_job(**batch_job_args)
        return response
    except sagemaker_client.exceptions.ResourceLimitExceeded as e:
        logger.error(f"Resource limit exceeded: {e}")
        raise
    except sagemaker_client.exceptions.ResourceInUse as e:
        logger.error(f"Resource in use: {e}")
        raise
    except Exception as e:
        logger.error(f"Error creating batch job: {e}")
        raise

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    s3_client = boto3.client('s3')
    sagemaker_client = boto3.client('sagemaker')
    
     # Set datetime
    offset = datetime.timedelta(hours=2)
    utc_now = datetime.datetime.utcnow()
    madrid_now = utc_now + offset
    date = madrid_now.strftime('%m-%d-%H%M%S')
    
    source_bucket = event.get('source_bucket', 'qloudy-xgboost-demo')
    model_package_group_name = event.get('model_package_group_name')
    batch_config_key = event.get('batch_config_key')
    debug_ = event.get('debug_', False)
    
    try:
        config_data = get_config_data(s3_client, source_bucket, batch_config_key)
        instancetype = config_data['InstanceType']
        s3_output_path = config_data['S3OutputPath']
        data_source = config_data['DataSourceS3Uri']
        model_image = config_data['TrainingImage']
        role = config_data['RoleArn']
        
        # Get the last batch data .csv with no target
        prefix_nt = data_source.split(f's3://{source_bucket}/')[-1] + 'batch_nt'
        last_batch_nt = get_last_modified(prefix_nt, source_bucket, s3_client)
        logger.info(f'Using last batch df: {last_batch_nt}')
 
    except KeyError as e:
        missing_field = e.args[0]
        logger.error(f"Missing required field in the configuration file: {missing_field}")
        return {
            'statusCode': 400,
            'body': f'Missing required field in the configuration file: {missing_field}'
        }
    except Exception as e:
        logger.error(f"Error fetching or parsing the configuration file from S3: {e}")
        return {
            'statusCode': 500,
            'body': 'Error fetching or parsing the configuration file from S3.'
        }
    
    # Get the 'Approved' or production model from the Model Package Group
    approved_package = sagemaker_client.list_model_packages(
                        ModelPackageGroupName=model_package_group_name,
                        ModelApprovalStatus='Approved')

    if not approved_package['ModelPackageSummaryList']:
        logger.error('No approved model packages found.')
        return  {
            'statusCode': 500,
            'body': 'Error: no approved model packages found.'
        }
    
    approved_arn = approved_package['ModelPackageSummaryList'][0]['ModelPackageArn']
    approved_model_desc = sagemaker_client.describe_model_package(ModelPackageName=approved_arn)
    approved_model_url = approved_model_desc['InferenceSpecification']['Containers'][0]['ModelDataUrl']
    approved_model_package_version = approved_model_desc['ModelPackageVersion']

    logger.info(f'Approved model package: {approved_arn}, version {approved_model_package_version}')

    # Create the model object from the 'Approved' model
    model_name = f'wine-quality-model-v-{approved_model_package_version}-{date}'
    create_model_args = {
        "ModelName": model_name,
        "PrimaryContainer": {
            "Image": model_image,
            "ModelDataUrl": approved_model_url
        },
        "ExecutionRoleArn": role,
        "EnableNetworkIsolation": False
    }

    try:
        response = sagemaker_client.create_model(**create_model_args)
        model_arn = response["ModelArn"]
        logger.info(f'Model object: {model_name} created successfully from "Approved" model package: {approved_arn}')
    except Exception as e:
        logger.error(f'Error creating model: {str(e)}')
        return  {
            'statusCode': 500,
            'body': f'Error creating model: {str(e)}'
        }
        
    # Now create the batch transform job definition
    
    batch_transform_job_name = f"wine-quality-predictor-{date}"
    batch_transform_job_args = {
        'TransformJobName' : batch_transform_job_name,
        'ModelName' : model_name,
        'TransformInput' : {
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': f's3://{source_bucket}/{last_batch_nt}'
                }
            },
            'ContentType': 'text/csv',
            'SplitType': 'Line'
        },
        'TransformOutput' : {
            'S3OutputPath': s3_output_path
        },
        'TransformResources' : {
            'InstanceType': instancetype,
            'InstanceCount': 1
        }
    }
    
    if not debug_:
        logger.info("Debug mode is disabled, creating batch job creation.")
        try:
            response = create_batch_job(sagemaker_client, batch_transform_job_args)
            logger.info(f"Successfully created the batch job: {batch_transform_job_name}")
        except Exception as e:
            logger.error(f"Error creating batch job: {e}")
            return {
                'statusCode': 500,
                'body': 'Error creating batch job.'
            }
    else:
        logger.info("Debug mode is enabled, skipping batch job creation.")
        training_job_name = ''
        response = None
    
    results = {
        'BatchJobName':batch_transform_job_name,
        'ApprovedModelPackageArn':approved_arn,
        'ApprovedModelPackageUrl':approved_model_url,
        'ModelName':model_name,
        'source_bucket':source_bucket
    }
    
    return results