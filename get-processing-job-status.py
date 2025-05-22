""" Function to retrieve processing status and check its completion
"""

import boto3
import logging
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sm_client = boto3.client('sagemaker')

# read hpo status and pass it on to verify its completion
def lambda_handler(event, context):
    
    # print(event)
    job_name = event.get('ProcessingJobName',None)
    source_bucket = event.get('source_bucket')
    debug_ = event.get('debug_', False)
   

    #Query boto3 API to check proces status.
    if not debug_ and job_name:
        try:
            response = sm_client.describe_processing_job(ProcessingJobName=job_name)
            status = response['ProcessingJobStatus']
            logger.info(f"Processing job:{job_name} has status:{response['ProcessingJobStatus']}.")
            
        except Exception as e:
            response = ('Failed to read processing status!'+ 
                        ' The processing job may not exist or the job name may be incorrect.'+ 
                        ' Check SageMaker to confirm the job name.')
            logger.error(e)
            logger.info(f'{response}. Attempted to read job name: {job_name}.')

    else:
        job_name = ''
        status = ''

    return {
        'ProcessingJobName':job_name,
        'Status':status,
        'source_bucket':source_bucket
    }