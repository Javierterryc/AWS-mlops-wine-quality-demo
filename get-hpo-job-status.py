""" Function to retrieve HPO status and check its completion
"""

import boto3
import logging
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sm_client = boto3.client('sagemaker')

# read hpo status and pass it on to verify its completion
def lambda_handler(event, context):
    source_bucket = event.get('source_bucket')
    HPOJobName = event.get('HPOJobName') 
    debug_ = event.get('debug_', False)
    
    if not debug_:
        try:
            response = sm_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName = HPOJobName)
            status = response['HyperParameterTuningJobStatus']
            logger.info(f"HPO job: '{HPOJobName}'' has status: '{status}''.")
        except Exception as e:
            response = ('Failed to read HPO status!'+ 
                        ' The HPO job may not exist or the name may be incorrect.'+ 
                        ' Check SageMaker to confirm the job name.')
            logger.error(e)
            logger.warning(f'{response} Attempted to read job name: {job_name}.')
    else:
        status = ''

    results = {
        'source_bucket':source_bucket,
        'HPOJobName':HPOJobName,
        'Status':status
    }
    return results