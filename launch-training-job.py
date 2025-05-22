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

    Retreives training config file
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

def get_last_modified(prefix, bucket, s3_client) -> Dict[str, Any]:
    '''
    This function lists the objects in an S3 bucket with a given prefix, identifies the most recent
    file, and then downloads and parses it into a Python dictionary. 
    
    Retrieves the last modified/added json 
    '''
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        contents = response['Contents']
        contents_sorted = sorted(contents, key=lambda obj: obj['LastModified'])
        last_object_key = contents_sorted[-1]['Key']
        metadata_object = s3_client.get_object(Bucket=bucket, Key=last_object_key)
        metadata_json = json.loads(metadata_object['Body'].read().decode('utf-8'))
        return metadata_json, last_object_key
        
    except s3_client.exceptions.NoSuchKey:
        logger.error(f"Processing metadata file not found with prefix: {prefix}")
        raise
    except Exception as e:
        logger.error(f"Error fetching processing metadata from S3: {e}")
        raise

def create_training_job(sagemaker_client, training_job_args: Dict[str, Any]) -> Dict[str, Any]:
    '''
    Function to create and launch a training job given the sagemaker client and the training job arguments
    '''
    
    try:
        response = sagemaker_client.create_training_job(**training_job_args)
        return response
    except sagemaker_client.exceptions.ResourceLimitExceeded as e:
        logger.error(f"Resource limit exceeded: {e}")
        raise
    except sagemaker_client.exceptions.ResourceInUse as e:
        logger.error(f"Resource in use: {e}")
        raise
    except Exception as e:
        logger.error(f"Error creating training job: {e}")
        raise

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    s3_client = boto3.client('s3')
    sagemaker_client = boto3.client('sagemaker')

    source_bucket = event.get('source_bucket')
    eval_metric = event.get('eval_metric', None)
    training_config_key = event.get('source_config_key')
    default_hyp_config_key = event.get('default_hyp_config_key')
    best_hyp_config_prefix = event.get('best_hyp_config_prefix')
    model_package_group_name = event.get('model_package_group_name')
    debug_ = event.get('debug_', False)
    
    if eval_metric is None:
        logger.error("Evaluation metric ('eval_metric') not provided.")
        return {
            'statusCode': 400,
            'body': "Evaluation metric ('eval_metric') not provided."
        }
    
    try:
        config_data = get_config_data(s3_client, source_bucket, training_config_key)
        training_image = config_data['TrainingImage']
        role = config_data['RoleArn']
        instancetype = config_data['InstanceType']
        training_job_output_path = config_data['S3OutputPath']
       
        response = s3_client.list_objects_v2(Bucket=source_bucket, Prefix=best_hyp_config_prefix)
        
        # First, try to find a hyp json from a previous HPO. If there isn't, then use default hyperparameters
        if not 'Contents' in response:
            logger.info('There are no hyperparameters metadata')
            hyp_data = get_config_data(s3_client, source_bucket, default_hyp_config_key)
            hyperparameters = hyp_data['HyperParameters']
            hyperparameters['eval_metric'] = eval_metric
            logger.info('Using default hyperparameters')
        else:
            # Get the last hyperparameters
            logger.info('There are hyperparameters metadata')
            hyp_data, last_hyp_object_key = get_last_modified(best_hyp_config_prefix, source_bucket, s3_client)
            hyperparameters = hyp_data['TrainingJobMetadata']['TunedHyperParameters']
            hyperparameters['objective'] = 'binary:logistic'
            hyperparameters['eval_metric'] = eval_metric
            
            last_hyp_object = last_hyp_object_key.split('/')[-1]
            logger.info(f'Using best hyperparameters from last hyperparameters json: {last_hyp_object}')
            
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

    # Set datetime
    offset = datetime.timedelta(hours=2)
    utc_now = datetime.datetime.utcnow()
    madrid_now = utc_now + offset
    date = madrid_now.strftime('%m-%d-%H%M%S')
    training_job_name = "wine-quality-estimator-" + date
    
    processing_metadata_prefix = 'wine-quality-project/pipeline-metadata/processing-job-metadata'
    
    # Retreive the last df to launch the training job
    try:
        processing_metadata_json, last_processing_key = get_last_modified(processing_metadata_prefix, source_bucket, s3_client)
        last_train_df_path = processing_metadata_json['DatasetProperties']['train_uri']
        last_test_df_path = processing_metadata_json['DatasetProperties']['test_uri']
        
        last_processing_object = last_processing_key.split('/')[-1]
        logger.info(f'Using df from last processing json: {last_processing_object}')
    except Exception as e:
        logger.error(f"Error fetching processing metadata from S3: {e}")
        return {
            'statusCode': 500,
            'body': 'Error fetching processing metadata from S3.'
        }
  
    # Define training job args
    
    training_job_args = {
        'TrainingJobName': training_job_name,
        'HyperParameters': hyperparameters,
        'AlgorithmSpecification': {
            'TrainingImage': training_image,
            'TrainingInputMode': 'File',
            # 'EnableSageMakerMetricsTimeSeries': False
        },
        'RoleArn': role,
        'InputDataConfig': [
            {
                'ChannelName': 'train',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': last_train_df_path,
                        'S3DataDistributionType': 'FullyReplicated',
                    }
                },
                'ContentType': 'text/csv',
                'CompressionType': 'None'
            },
            {
                'ChannelName': 'validation',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': last_test_df_path,
                        'S3DataDistributionType': 'FullyReplicated',
                    }
                },
                'ContentType': 'text/csv',
                'CompressionType': 'None'
            },
        ],
        'OutputDataConfig': {'S3OutputPath': training_job_output_path},
        'ResourceConfig': {'InstanceType': instancetype, 
                           'InstanceCount': 1, 
                           'VolumeSizeInGB': 5},
        'StoppingCondition': {'MaxRuntimeInSeconds': 3600,
                              'MaxWaitTimeInSeconds': 3600},
        'EnableNetworkIsolation': False,
        'EnableManagedSpotTraining': True
    }
    
    if not debug_:
        try:
            response = create_training_job(sagemaker_client, training_job_args)
            logger.info(f"Successfully created the training job: {training_job_name}")
        except Exception as e:
            logger.error(f"Error creating training job: {e}")
            return {
                'statusCode': 500,
                'body': 'Error creating training job.'
            }
    else:
        logger.info("Debug mode is enabled, skipping training job creation.")
        training_job_name = ''
        response = None
    
    results = {
        'TrainingJobName': training_job_name,
        'eval_metric': eval_metric,
        'source_bucket':source_bucket,
        'model_package_group_name': model_package_group_name
    }
    
    return results
