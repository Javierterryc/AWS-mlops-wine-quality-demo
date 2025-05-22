import json
from datetime import datetime
import pandas as pd

def lambda_handler(event,context):
    # Extract bucket name and file key from S3 path
    
    file_path = event.get('file_path')
    bucket_name = file_path.split('/')[2]  # Extracting bucket name from s3 path
    file_key = '/'.join(file_path.split('/')[3:])  # Extracting file key from s3 path

    df = pd.read_csv(file_path)
    
    s3 = boto3.client('s3')  # Uses configured credentials
    
    # Get object metadata to fetch last modified date
    response = s3.head_object(Bucket=bucket_name, Key=file_key)
    last_modified = response['LastModified'].strftime("%Y-%m-%d %H:%M:%S")
    
    # Basic Metadata
    metadata = {}
    metadata['dataset_name'] = file_key.split('/')[-1]
    metadata['dataset_source'] = file_path
    metadata['creation_date'] = last_modified
    
    # Structural Metadata
    if 'Structural_Metadata' not in metadata:
        metadata['Structural_Metadata'] = {}
        
    metadata['Structural_Metadata']['schema'] = df.dtypes.apply(lambda x: x.name).to_dict()
    metadata['Structural_Metadata']['num_columns'] = df.shape[1]
    metadata['Structural_Metadata']['num_rows'] = df.shape[0]
    metadata['Structural_Metadata']['file_format'] = 'CSV'
    
    # Statistical Metadata
    basic_stats = df.describe().to_dict()
    metadata['basic_statistics'] = basic_stats
    
    # Missing Values
    missing_values = df.isnull().sum().to_dict()
    metadata['missing_values'] = missing_values
    
    # Unique Values
    unique_values = df.nunique().to_dict()
    metadata['unique_values'] = unique_values
    
    # Data Quality
    metadata['duplicates'] = df.duplicated().sum()
    
    # Technical Metadata
    metadata['storage_location'] = file_path
    
    return metadata