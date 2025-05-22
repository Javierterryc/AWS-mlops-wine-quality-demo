import boto3
import logging
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sm_client = boto3.client('sagemaker')

# read hpo status and pass it on to verify its completion
def lambda_handler(event, context):
    
    # print(event)
    source_bucket = event.get('source_bucket')
    eval_metric = event.get('eval_metric')
    model_package_group_name = event.get('model_package_group_name')
    job_name = event["TrainingJobName"]
    debug_ = event.get('debug_', False)
   
    #Query boto3 API to check proces status.
    if not debug_:
        try:
            response = sm_client.describe_training_job(TrainingJobName=job_name)
            status = response['TrainingJobStatus']
            logger.info("Training job:{} has status:{}.".format(job_name,status))
            
    
        except Exception as e:
            response = ('Failed to read training status!'+ 
                        ' The training job may not exist or the job name may be incorrect.'+ 
                        ' Check SageMaker to confirm the job name.')
            print(e)
            print('{} Attempted to read job name: {}.'.format(response, job_name))
    else:
        status = ''

    results = {
        'source_bucket':source_bucket,
        'eval_metric': eval_metric,
        'TrainingJobName':job_name,
        'Status':status,
        'model_package_group_name':model_package_group_name
    }
    
    return results