import json
import logging
import boto3
from botocore.exceptions import BotoCoreError,ClientError
import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Set datetime
offset = datetime.timedelta(hours=2)
utc_now = datetime.datetime.utcnow()
madrid_now = utc_now + offset
date = madrid_now.strftime('%m-%d-%H%M%S')

s3_client = boto3.client('s3')
sagemaker_client = boto3.client('sagemaker')

def create_model_package(source_bucket,metadata_key, model_package_group_name, model_status='PendingManualApproval'):
    '''
        Function that creates a model package in a model package group from a metadata json containing metadata of a training job
    '''
    try:
        # Get last training job metadata JSON file from S3
        response = s3_client.get_object(Bucket=source_bucket, Key=metadata_key)
        training_metadata = response['Body'].read().decode('utf-8')
        training_metadata = json.loads(training_metadata)
        
        metrics = training_metadata['ModelRegistry']['ModelMetrics']
        image = training_metadata['TrainingJobMetadata']['TrainingImage']
        model_data_url = training_metadata['ModelRegistry']['ModelDataUrl']
        training_job_date = training_metadata['TrainingJobMetadata']['TrainingEndTime']
        model_status='PendingManualApproval'

        # Prepare the package creation arguments
        package_args = {
            'ModelPackageGroupName': model_package_group_name,
            'ModelPackageDescription': '.',
            'InferenceSpecification': {
                'Containers': [
                    {
                        'Image': image,
                        'ModelDataUrl': model_data_url
                    }
                ],
                'SupportedContentTypes': ['text/csv'],
                'SupportedResponseMIMETypes': ['text/csv'],
            },
            'ModelApprovalStatus': model_status,
            'CustomerMetadataProperties': {
                'TrainingJobDate': training_job_date,
                'Model_version': '.'
            }
        }

        for metric in metrics:
            name = metric['MetricName']
            value = metric['Value']
            package_args['CustomerMetadataProperties'][str(name)] = str(value)

        # Create the new model package
        response = sagemaker_client.create_model_package(**package_args)
        model_package_arn = response['ModelPackageArn']
        
        # Get model version
        response = sagemaker_client.describe_model_package(ModelPackageName=model_package_arn)
        model_version = response['ModelPackageVersion']
        
        # Update the model package with the correct model version
        sagemaker_client.update_model_package(
            ModelPackageArn=model_package_arn,
            CustomerMetadataProperties={
                'Model_version': f'1.0.{model_version}'
            }
        )
        
        return model_package_arn

    except (BotoCoreError, ClientError) as error:
        logger.error(f"An error occurred in the create_model_package function: {error}")
        raise
    except KeyError as e:
        logger.error(f"Key error: {e}")
        raise
    except Exception as e:
        logger.error(f"Exception error: {e}")
        raise

def get_model_metric(model_package_arn, metric_name):
    try:
        response = sagemaker_client.describe_model_package(ModelPackageName=model_package_arn)
        return response['CustomerMetadataProperties'][metric_name]
    except (KeyError, BotoCoreError, ClientError) as error:
        logger.error(f"An error occurred while retrieving the metric: {error}")
        raise

def lambda_handler(event, context):
    
    source_bucket = event.get('source_bucket')
    training_metadata_name = event.get('TrainingMetadataJson', None)
    eval_metric = event.get('eval_metric', None)
    model_package_group_name = event.get('model_package_group_name')
    metric_name = f'validation:{eval_metric}'
    metadata_key = f'wine-quality-project/pipeline-metadata/training-job-metadata/{training_metadata_name}'
    
    try:
        response = sagemaker_client.list_model_package_groups()
        model_package_group_names = [group['ModelPackageGroupName'] for group in response['ModelPackageGroupSummaryList']]
    
        if model_package_group_name in model_package_group_names:
            # Model Package Group already exists
            logger.info(f"Model Package Group '{model_package_group_name}' already exists.")
        else:
            # Create the Model Package Group
            logger.info(f"Creating Model Package Group with name: '{model_package_group_name}'")
            sagemaker_client.create_model_package_group(
                ModelPackageGroupName=model_package_group_name
            )
            logger.info(f"Model Package Group '{model_package_group_name}' created successfully.")
    
    except ClientError as e:
        logger.error(f"An error occurred while creating Model Package Group: {e}")
        raise e
    
    try:
        # Create a model package from the last training job metadata
        last_training_job_model_package_arn = create_model_package(source_bucket,metadata_key, model_package_group_name)
    
        # Retrieve the ARN of the approved model package
        try:
            approved_package = sagemaker_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus='Approved')
            num_approved_packages = len(approved_package['ModelPackageSummaryList'])
    
            if num_approved_packages == 1:
                logger.info("At least one approved model package exists.")
                production_model_package_arn = approved_package['ModelPackageSummaryList'][0]['ModelPackageArn']
            elif num_approved_packages == 0:
                production_model_package_arn = None
                logger.info("There are no models in production.")
            else:
                production_model_package_arn = None
                logger.warning("There are more than 1 models in production.")
                return {
                    'statusCode':500,
                    'body':json.dumps({
                        'message': 'There are more than 1 models in production'
                    })
                }
                
        except (BotoCoreError, ClientError) as error:
            logger.error(f"An error occurred in the get_approved_model_package_arn function: {error}")
        
        # If there aren't any models in production, the last training job model is set to production
        if not production_model_package_arn:
            sagemaker_client.update_model_package(
                ModelPackageArn=last_training_job_model_package_arn,
                ModelApprovalStatus='Approved'
            )
            logger.info('First model package approved as there are no existing approved models.')
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'First model package approved as there are no existing approved models.',
                    'New best model package:': last_training_job_model_package_arn
                })
            }
    
        # Retrieve model metrics
        try:
            production_model_metric = get_model_metric(production_model_package_arn, metric_name)
        except Exception as e:
            logger.error(f"An error occurred with the production_model_metric: {e}")
            return {
                'statusCode': 500,
                'body': json.dumps({'message': f'Production_model_metric error: {str(e)}'})
            }
        try:   
            last_training_job_metric = get_model_metric(last_training_job_model_package_arn, metric_name)
        except Exception as e:
            logger.error(f"An error occurred with the last_training_job_metric: {e}")
            return {
                'statusCode': 500,
                'body': json.dumps({'message': f'Last_training_job_metric error: {str(e)}'})
            }
    
        maximize_metrics = ['accuracy','auc','f1','map','ndcg']
        maximize = any(metric in metric_name for metric in maximize_metrics)
        
        if maximize:
            logger.info('Comparing maximum metric')
            # Compare the metrics
            if float(production_model_metric) > float(last_training_job_metric):
                best_model = 'production'
            else:
                best_model = 'new'
        else:
            logger.info('Comparing minimum metric')
            if float(production_model_metric) < float(last_training_job_metric):
                best_model = 'production'
            else:
                best_model = 'new'
    
        if best_model == 'production':
            # Update last training job model package from 'PendingManualApproval' to 'Rejected'
            sagemaker_client.update_model_package(
                ModelPackageArn=last_training_job_model_package_arn,
                ModelApprovalStatus='Rejected'
            )
            logger.info('Production model remains the best')
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'Production model remains the best'}),
                'Production model package:': production_model_package_arn
            }
        else:
            sagemaker_client.update_model_package(
                ModelPackageArn=last_training_job_model_package_arn,
                ModelApprovalStatus='Approved'
            )
            sagemaker_client.update_model_package(
                ModelPackageArn=production_model_package_arn,
                ModelApprovalStatus='Rejected'
            )
            logger.info('New model from last training job is the best now')
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'New model from last training job is the best now'}),
                'New best model package:': last_training_job_model_package_arn
            }
    
    except (KeyError, BotoCoreError, ClientError) as error:
        logger.error(f"An error occurred during model comparison: {error}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': f'Error: {str(error)}'})
        }
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': f'Unexpected error: {str(e)}'})
        }