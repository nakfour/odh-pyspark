import pyspark



print("Start")
#on openshift
#spark = SparkSession.builder.appName("mobileanalytics").config("spark.mongodb.input.uri", "mongodb://admin:admin@mongodb/sampledb.bikerental").config("spark.mongodb.output.uri", "mongodb://admin:admin@mongodb/sampledb.bikerental").getOrCreate()
#Set the configuration
conf = pyspark.SparkConf().setAppName('ODH PySpark').setMaster('spark://' + os.environ['OSHINKO_CLUSTER_NAME'] + ':7077')

#Set the Spark cluster connection
sc = pyspark.SparkContext.getOrCreate(conf)


#Get the SQL context
sqlContext = pyspark.SQLContext(sc)
