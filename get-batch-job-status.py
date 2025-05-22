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
    
    # print(event)
    job_name = event.get('BatchJobName')
    source_bucket = event.get('source_bucket')
    debug_ = event.get('debug_', False)
   

    #Query boto3 API to check proces status.
    if not debug_:
        try:
            response = sm_client.describe_transform_job(TransformJobName=job_name)
            status = response['TransformJobStatus']
            logger.info("Transform job:{} has status:{}.".format(job_name,status))
            
    
        except Exception as e:
            response = ('Failed to read training status!'+ 
                        ' The training job may not exist or the job name may be incorrect.'+ 
                        ' Check SageMaker to confirm the job name.')
            print(e)
            print('{} Attempted to read job name: {}.'.format(response, job_name))
    else:
        status = ''

    results = {
        'BatchJobName':job_name,
        'Status':status,
        'source_bucket':source_bucket
    }
    
    return results