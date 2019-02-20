import pyspark
import os
from pyspark.sql import SparkSession, SQLContext
from pyspark.context import SparkContext

print("Start")
#on openshift
spark = SparkSession.builder.appName("odh-pyspark").getOrCreate()

#Reading from local file
#df=spark.read.csv('sample.csv',header=True)
#print(df.head())

#Reading CEPH setup
accessKey= os.environ['ACCESS_KEY_ID']
secretKey= os.environ['SECRET_ACCESS_KEY']
endpointUrl= os.environ['S3_ENDPOINT_URL']
print(endpointUrl)
#s3Bucket= os.environ['S3BUCKET']

#Set the Hadoop configurations to access Ceph S3
hadoopConf=spark.sparkContext._jsc.hadoopConfiguration()
hadoopConf.set("fs.s3a.access.key", accessKey) 
hadoopConf.set("fs.s3a.secret.key", secretKey) 
hadoopConf.set("fs.s3a.endpoint", endpointUrl) 

#Get the SQL context
sqlContext = pyspark.SQLContext(sc)

#feedbackFile = sqlContext.read.option("sep", "\t").csv("s3a://" + s3Bucket + "/datasets/sentiment_data.tsv", header=True)

#feedbackFile.show()

#Stop the spark cluster
spark.stop()
