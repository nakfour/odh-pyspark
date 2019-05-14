# This program just collects data and stores them in .csv file in S3 storage
# At this point we just read the file from local folder and upload to S3
# This can be changed later to add filtering and collecting data from databases or other stores

import os
import boto3


print("Start")

#Reading CEPH setup
accessKey= os.environ['ACCESS_KEY_ID']
secretKey= os.environ['SECRET_ACCESS_KEY']
endpointUrl= os.environ['S3_ENDPOINT_URL']
print(endpointUrl)
#s3Bucket= os.environ['S3BUCKET']


# Create an S3 client
s3 = boto3.client(service_name='s3',aws_access_key_id=accessKey, aws_secret_access_key=secretKey, endpoint_url=endpointUrl)
s3.create_bucket(Bucket='OPEN')


# Upload to Rook/Ceph in bucket Open and key uploaded/creditcard-sample10k.csv
key = "uploaded/creditcard-sample10k.csv"
s3.upload_file(Bucket='OPEN', Key=key, Filename="creditcard-sample10k.csv")
s3.list_objects(Bucket='OPEN')
