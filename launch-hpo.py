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
    
    Retreives HPO config file
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

def get_processing_metadata(s3_client, bucket: str, prefix: str) -> Dict[str, Any]:
    '''
    This function lists the objects in an S3 bucket with a given prefix, identifies the most recent
    file, and then downloads and parses it into a Python dictionary. 
    
    Retrieves the last datasets for the HPO 
    '''
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        processing_source_config_key = response['Contents'][-1]['Key']
        processing_metadata_object = s3_client.get_object(Bucket=bucket, Key=processing_source_config_key)
        processing_metadata_json = json.loads(processing_metadata_object['Body'].read().decode('utf-8'))
        return processing_metadata_json
    except s3_client.exceptions.NoSuchKey:
        logger.error(f"Processing metadata file not found with prefix: {prefix}")
        raise
    except Exception as e:
        logger.error(f"Error fetching processing metadata from S3: {e}")
        raise

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    
    s3_client = boto3.client('s3')
    sagemaker_client = boto3.client('sagemaker')
    
    source_bucket = event.get('source_bucket')
    source_config_key = event.get('source_config_key')
    objective_input = event.get('objective_input')
    debug_ = event.get('debug_', False)
    
    try:
        config_data = get_config_data(s3_client, source_bucket, source_config_key)
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
    
    
    # Tuning job config. It follows the hpo_config_file.json created in the Jupyter Notebook
    Strategy=config_data['Strategy']

    ResLimits={}
    ResLimits["MaxNumberOfTrainingJobs"]=int(config_data['ResourceLimits']['MaxNumberOfTrainingJobs'])
    ResLimits["MaxParallelTrainingJobs"]=int(config_data['ResourceLimits']['MaxParallelTrainingJobs'])
    
    Objective={}
    
    # If an objective input isn't specified in the payload, it will default to: "type" : "minimize", "objective" : "validation:logloss",
    # as it is defined in the HPO config file uploaded in S3.
    
    if not objective_input:
        Objective['Type'] = config_data['HyperParameterTuningJobObjective']['Type']
        Objective['MetricName'] = config_data['HyperParameterTuningJobObjective']['MetricName']
        logger.info(f'Using default objective definition from HPO config file: {Objective}')
    else:
        objective = json.loads(objective_input)
        Objective['Type'] = objective['Type']
        Objective['MetricName'] = objective['MetricName']
        logger.info(f'Using input objective specified in the event: {objective}')

    ParamRanges = {
        "CategoricalParameterRanges": [],
        "IntegerParameterRanges": [],
        "ContinuousParameterRanges": []
    }
    
    
    # Convert integer ranges to strings
    integer_ranges = config_data['ParameterRanges'].get('IntegerParameterRanges', [])
    for range_dict in integer_ranges:
        ParamRanges["IntegerParameterRanges"].append({
            "Name": range_dict.get("Name"),
            "MinValue": str(range_dict.get("MinValue")),
            "MaxValue": str(range_dict.get("MaxValue")),
            "ScalingType":str(range_dict.get("ScalingType"))
        })

    # Convert continuous ranges to strings
    continuous_ranges = config_data['ParameterRanges'].get('ContinuousParameterRanges', [])
    for range_dict in continuous_ranges:
        ParamRanges["ContinuousParameterRanges"].append({
            "Name": range_dict.get("Name"),
            "MinValue": str(range_dict.get("MinValue")),
            "MaxValue": str(range_dict.get("MaxValue")),
            "ScalingType":str(range_dict.get("ScalingType"))
        })

    tuning_job_config={
        "Strategy":Strategy,
        "ResourceLimits":ResLimits,
        "HyperParameterTuningJobObjective":Objective,
        "ParameterRanges":ParamRanges
    }
    
    processing_metadata_prefix = 'wine-quality-project/pipeline-metadata/processing-job-metadata'
    try:
        # Retreive the last df created 
        processing_metadata_json = get_processing_metadata(s3_client, source_bucket, processing_metadata_prefix)
        last_hpo_train_df_path = processing_metadata_json['DatasetProperties']['hpo_train_uri']
        last_hpo_test_df_path = processing_metadata_json['DatasetProperties']['hpo_test_uri']
    except Exception as e:
        logger.error(f"Error fetching processing metadata from S3: {e}")
        return {
            'statusCode': 500,
            'body': 'Error fetching processing metadata from S3.'
        }
    
    TrainingImage=config_data['TrainingImage']
    S3OutputPath=config_data['S3OutputPath']
    InstanceType=config_data['InstanceType']
    RoleArn=config_data['RoleArn']
    
    training_job_definition = {
        "AlgorithmSpecification": {
          "TrainingImage": TrainingImage,
          "TrainingInputMode": "File",
          "MetricDefinitions": [
               {'Name': 'train:mae',
                'Regex': '.*\\[[0-9]+\\].*#011train-mae:([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*'},
               {'Name': 'validation:aucpr',
                'Regex': '.*\\[[0-9]+\\].*#011validation-aucpr:([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*'},
               {'Name': 'validation:f1_binary',
                'Regex': '.*\\[[0-9]+\\].*#011validation-f1_binary:([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*'},
               {'Name': 'validation:mae',
                'Regex': '.*\\[[0-9]+\\].*#011validation-mae:([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*'},
               {'Name': 'validation:logloss',
                'Regex': '.*\\[[0-9]+\\].*#011validation-logloss:([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*'},
               {'Name': 'validation:f1',
                'Regex': '.*\\[[0-9]+\\].*#011validation-f1:([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*'},
               {'Name': 'train:accuracy',
                'Regex': '.*\\[[0-9]+\\].*#011train-accuracy:([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*'},
               {'Name': 'validation:recall',
                'Regex': '.*\\[[0-9]+\\].*#011validation-recall:([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*'},
               {'Name': 'validation:precision',
                'Regex': '.*\\[[0-9]+\\].*#011validation-precision:([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*'},
               {'Name': 'train:error',
                'Regex': '.*\\[[0-9]+\\].*#011train-error:([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*'},
               {'Name': 'validation:auc',
                'Regex': '.*\\[[0-9]+\\].*#011validation-auc:([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*'},
               {'Name': 'train:auc',
                'Regex': '.*\\[[0-9]+\\].*#011train-auc:([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*'},
               {'Name': 'validation:error',
                'Regex': '.*\\[[0-9]+\\].*#011validation-error:([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*'},
               {'Name': 'train:rmse',
                'Regex': '.*\\[[0-9]+\\].*#011train-rmse:([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*'},
               {'Name': 'train:logloss',
                'Regex': '.*\\[[0-9]+\\].*#011train-logloss:([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*'},
               {'Name': 'validation:accuracy',
                'Regex': '.*\\[[0-9]+\\].*#011validation-accuracy:([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*'}]
        },
        "InputDataConfig": [
          {
            "ChannelName": "train",
            "DataSource": {
              "S3DataSource": {
                "S3DataDistributionType": "FullyReplicated",
                "S3DataType": "S3Prefix",
                "S3Uri": last_hpo_train_df_path
              }
            },
            "ContentType": "text/csv",
            "CompressionType": "None"
          },
          {
            "ChannelName": "validation",
            "DataSource": {
              "S3DataSource": {
                "S3DataDistributionType": "FullyReplicated",
                "S3DataType": "S3Prefix",
                "S3Uri": last_hpo_test_df_path
              }
            },
            "ContentType": "text/csv",
            "CompressionType": "None"
          }
        ],
        "OutputDataConfig": {
          "S3OutputPath": S3OutputPath
        },
        "ResourceConfig": {
          "InstanceCount": 1,
          "InstanceType": InstanceType, # hard coded instance configuration
          "VolumeSizeInGB": 30
        },
        "RoleArn": RoleArn,
        "StoppingCondition": {
          'MaxRuntimeInSeconds': 6000,
          'MaxWaitTimeInSeconds': 8200
        },
        'EnableManagedSpotTraining': True
    }
    
    tuning_job_name = "wine-hpo-" + date
    
    if not debug_:
        try:
            response=sagemaker_client.create_hyper_parameter_tuning_job(HyperParameterTuningJobName = tuning_job_name,
                                           HyperParameterTuningJobConfig = tuning_job_config,
                                           TrainingJobDefinition = training_job_definition)
            
            logger.info(f"Successfully created the hpo job: {tuning_job_name}")
        except Exception as e:
            logger.error(f"Error creating hpo job: {e}")
            return {
                'statusCode': 500,
                'body': 'Error creating hpo job.'
            }
    else:
        logger.info("Debug mode is enabled, skipping hpo job creation.")
        tuning_job_name = ''
        response = None
    
    results = {
        'HPOJobName': tuning_job_name,
        'source_bucket': source_bucket
    }
    
    return results