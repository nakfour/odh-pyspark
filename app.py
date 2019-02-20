import pyspark
from pyspark.sql import SparkSession


print("Start")
#on openshift
spark = SparkSession.builder.appName("odh-pyspark").getOrCreate()
#df=spark.read.csv('sample.csv',header=True)
#print(df.head())

