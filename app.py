from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import unix_timestamp
from pyspark.sql.functions import year, month, dayofmonth, hour
from pyspark.sql.types import StringType, IntegerType, DoubleType
from pyspark.sql.functions import udf
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GeneralizedLinearRegression,LinearRegression
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pymongo import MongoClient
import pprint
import requests
import json
from bson import json_util
from bson.json_util import dumps
import random



def assign_tod(hr):
    #print(hr)
    times_of_day = {
    #'morning' : range(0,12),
    'morning' : range(0,12),
    #'lunch': range(12,14),
    'lunch': range(12,14),
    #'afternoon': range(14,18),
    'afternoon': range(14,18),
    #'evening': range (18,20),
    'evening': range (18,20),
    #'night': range(20,24)
    'night': range(20,24)
    }
    for k, v in times_of_day.iteritems():
        if hr in v:
            #print k
            return k

 

 
print("Start")
#on openshift
spark = SparkSession.builder.appName("mobileanalytics").config("spark.mongodb.input.uri", "mongodb://admin:admin@mongodb/sampledb.bikerental").config("spark.mongodb.output.uri", "mongodb://admin:admin@mongodb/sampledb.bikerental").getOrCreate()

#spark = SparkSession.builder.master("local").appName("mobileanalytics").config("spark.driver.bindAddress", "127.0.0.1").config("spark.mongodb.input.uri", "mongodb://127.0.0.1/sampledb.bikerental").config("spark.mongodb.output.uri", "mongodb://127.0.0.1/sampledb.bikerental").getOrCreate()
#spark.stop()


######## Predictive Analysis ##################
# We do this once on load of container at this time. It can be adapted to be dynamic later on ########
# 2013-07-01
fullbikerentaldf = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()
print(fullbikerentaldf.show())
fullbikerentaldf.printSchema()

############# heat map data: rental count, daypart, station id, lat, lon ##################################
heatmapdata= fullbikerentaldf.select(hour(unix_timestamp(fullbikerentaldf.starttime).cast('timestamp')).alias('starttimehour'),'startstationid', 'startstationlat', 'startstationlon').cache()
print("Heatmap")
print(heatmapdata.show())
heatmapdata.printSchema()

##### We need the daypart as integer to use it in the feature vector
def assign_tod_integer(hr):
    #print(hr)
    times_of_day = {
    #'morning' : range(0,12),
    0 : range(0,12),
    #'lunch': range(12,14),
    1: range(12,14),
    #'afternoon': range(14,18),
    2: range(14,18),
    #'evening': range (18,20),
    3: range (18,20),
    #'night': range(20,24)
    4: range(20,24)
    }
    for k, v in times_of_day.iteritems():
        if hr in v:
            #print k
            return k
func_udf_int = udf(assign_tod_integer, IntegerType())
dfheatmap = heatmapdata.withColumn('daypartInt',func_udf_int(heatmapdata['starttimehour']))


df3=dfheatmap.groupBy("daypartInt", "startstationid","startstationlat","startstationlon").agg(count("*"))
df3=df3.withColumn("startstationid", df3["startstationid"].cast("integer"))
df3=df3.withColumn("startstationlat",df3["startstationlat"].cast(DoubleType()))
df3=df3.withColumn("startstationlon",df3["startstationlon"].cast(DoubleType()))
df3 = df3.select(col("daypartInt").alias("daypartInt"),col("startstationid").alias("startstationid"),col("count(1)").alias("rentalcount"),col("startstationlat").alias("startstationlat"), col("startstationlon").alias("startstationlon"))

print("Showing rental count by station id and day part")
print(df3.show())
df3.printSchema()
print("Number of rows in df3")
print(df3.count())
df3.describe().show()

################################# Linear Regression Training #######################################
########################## Linear Regression #########################################################

 
assembler=VectorAssembler(inputCols=['startstationid', 'daypartInt', 'startstationlat', 'startstationlon'], outputCol="features")
output = assembler.transform(df3)
print("Assembled columns 'startstationid', 'daypartInt' to vector column 'features'")
output.select("features", "rentalcount").show(truncate=False)

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8,featuresCol=assembler.getOutputCol(), labelCol="rentalcount")
pipelineLG= Pipeline(stages=[assembler,lr])
modelLR = pipelineLG.fit(df3)


# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(modelLR.stages[-1].coefficients))
print("Intercept: %s" % str(modelLR.stages[-1].intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = modelLR.stages[-1].summary

print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

#### Prediction ###############################
# Get only stationid and daypart to get the predictions
testbikerentaldfSelect = df3.select(col("startstationid").alias("startstationid"),col("startstationlat").alias("startstationlat"), col("startstationlon").alias("startstationlon") ).orderBy('startstationid')
testbikerentaldfSelect3=testbikerentaldfSelect.groupBy('startstationid','startstationlat', 'startstationlon').agg(count("*"))

testbikerentaldfSelect3.show()
testbikerentaldfSelect3.printSchema()

### Adding the 4 daypart values to each station id 0,1,2,3,4,
testbikerentaldfSelect0=testbikerentaldfSelect3.withColumn("daypartInt", lit(0))
testbikerentaldfSelect1=testbikerentaldfSelect3.withColumn("daypartInt", lit(1))
testbikerentaldfSelect2=testbikerentaldfSelect3.withColumn("daypartInt", lit(2))
testbikerentaldfSelect3=testbikerentaldfSelect3.withColumn("daypartInt",lit(3))
testbikerentaldfSelect4=testbikerentaldfSelect3.withColumn("daypartInt", lit(4))

predictions0 = modelLR.transform(testbikerentaldfSelect0)
prediection0Select=predictions0.select(col("prediction").alias("rentalcount"),col("startstationlat").alias("startstationlat"), col("startstationlon").alias("startstationlon"))
print("Prediction")
#print(prediection0Select)
#prediection0Select.show()
#prediection0Select.describe().show()

predictions1 = modelLR.transform(testbikerentaldfSelect1)
prediection1Select=predictions1.select(col("prediction").alias("rentalcount"),col("startstationlat").alias("startstationlat"), col("startstationlon").alias("startstationlon"))
print("Prediction")
#print(prediection1Select)
#prediection1Select.show()
#prediection1Select.describe().show()

predictions2 = modelLR.transform(testbikerentaldfSelect2)
prediection2Select=predictions2.select(col("prediction").alias("rentalcount"),col("startstationlat").alias("startstationlat"), col("startstationlon").alias("startstationlon"))
print("Prediction")
#print(prediection2Select)
#prediection2Select.describe().show()

predictions3 = modelLR.transform(testbikerentaldfSelect3)
prediection3Select=predictions3.select(col("prediction").alias("rentalcount"),col("startstationlat").alias("startstationlat"), col("startstationlon").alias("startstationlon"))
print("Prediction")
#print(prediection3Select)
#prediection3Select.show()

predictions4 = modelLR.transform(testbikerentaldfSelect4)
prediection4Select=predictions4.select(col("prediction").alias("rentalcount"),col("startstationlat").alias("startstationlat"), col("startstationlon").alias("startstationlon"))
print("Prediction")
#print(prediection4Select)
#prediection4Select.show()

########################### End Predictive Analysis #########################################

    
def getDayStats():
    print("Getting Day stats")
    df = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()
    print("posting df")
    df.printSchema()
    print(df.show())
    #Convert starttime from string to timestamp
    #'yyyy-MM-dd HH:mm:ss'
    typedbikerentaldf= df.select(unix_timestamp(df.starttime).cast('timestamp').alias('starttimehour'),\
    'startstationid')\
    .cache()
    # we only need the hour to put rentals in buckets of morning, lunch, afternoon, evening by station id
    typedbikerentaldfhour= typedbikerentaldf.select(hour('starttimehour').alias('starttimehour'),\
    'startstationid')\
    .cache()
    print(typedbikerentaldfhour.show())
    typedbikerentaldfhour.printSchema()

    #create a function to return the string value of the time of day
    func_udf = udf(assign_tod, StringType())
    dfbuckets = typedbikerentaldfhour.withColumn('daypart',func_udf(typedbikerentaldfhour['starttimehour']))
    print("Serving Station Stats data")
    test=dfbuckets.groupBy("daypart").count()
    print("Printing group by daypart")
    print(test.show())
    resultlist=test.toJSON().collect()
    return resultlist
    
def getMobileOsStats():
    print("Getting Mobile OS stats")
    df = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()
    print("posting df")
    df.printSchema()
    print(df.show())
    ###################### Getting mobileos stats #######################
    mobiledata = df.select(col("mobileos").alias("mobileos"))
    mobiledata=mobiledata.groupBy("mobileos").agg(count("*"))
    mobiledata = mobiledata.select(col("mobileos").alias("mobileos"),col("count(1)").alias("mobileoscount"))
    print (mobiledata.show())
    resultlist=mobiledata.toJSON().collect()
    print(resultlist)
    return resultlist
    
################### app Web Server #####################
app = Flask(__name__)
#CORS(app)
#CORS(app, resources=r'/*', headers='Content-Type')

@app.route("/")
def mainRoute():
    print("Serving /")
    return render_template("index.html")

@app.route("/getstationstats")
def dataRoute():
    results=getDayStats()
    print(results)
    json_results = json.dumps(results)
    print(json_results)
    return json_results

@app.route("/getmobileosstats")
def mobiledataRoute():
    print("Serving Mobile OS")
    results=getMobileOsStats()
    print(results)
    json_results = json.dumps(results)
    print(json_results)
    return json_results
    
@app.route("/gettouchdata")
def touchData():
    print("Serving Touch Data")
    ################### Reading Touch data from database ########################
    print("Reading Spark Touch Data");
    touchdata = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("uri","mongodb://admin:admin@mongodb/sampledb.touchdata").load()
    print(touchdata.show())
    touchdata.printSchema()
    resultlist=touchdata.toJSON().collect()
    print(resultlist)
    json_results = json.dumps(resultlist)
    print(json_results)
    return json_results   

#.defer(d3.json, "<Insert-url>&metric_name=hits&since=2017-07-01&period=year&granularity=month&skip_change=true")  
# had to do it this way because CORS is not enabled by s-scale by default
@app.route("/gethits")
def hitsRoute():
    print("Getting 3-scale hits")
    response=requests.get("<Insert-url>&metric_name=hits&since=2017-07-01&period=year&granularity=month&skip_change=true")
    print(response.status_code)
    print(response.headers)
    print(response.content)
    return (response.content)  

@app.route("/getpoststartrental")
def startRentalRoute():
    print("Getting 3-scale startrental")
    response=requests.get("<Insert-url>&metric_name=poststartrental&since=2017-07-01&period=year&granularity=month&skip_change=true")
    print(response.status_code)
    print(response.headers)
    print(response.content)
    return (response.content) 

@app.route("/getpoststoprental")
def stopRentalRoute():
    print("Getting 3-scale stoprental")
    response=requests.get("<Insert-url>&metric_name=poststoprental&since=2017-07-01&period=year&granularity=month&skip_change=true")
    print(response.status_code)
    print(response.headers)
    print(response.content)
    return (response.content) 
    
@app.route("/getstationdaytime")
def stationdaytimeRoute():
    print("Station id by daytime rentals")
    #We need lat lon and predictions (and daytime later)
    resultlist=df3.toJSON().collect()
    #print(resultlist)
    json_results = json.dumps(resultlist)
    return json_results
    
@app.route("/getstationdaytimemorning")
def stationdaytimeRouteMorning():
    print("Station id by daytime rentals Morning")
    #We need lat lon and predictions (and daytime later)
    resultlist=prediection0Select.toJSON().collect()
    #print(resultlist)
    json_results = json.dumps(resultlist)
    return json_results

@app.route("/getstationdaytimelunch")
def stationdaytimeRouteLunch():
    print("Station id by daytime rentals Lunch")
    #We need lat lon and predictions (and daytime later)
    resultlist=prediection1Select.toJSON().collect()
    #print(resultlist)
    json_results = json.dumps(resultlist)
    return json_results     

@app.route("/getstationdaytimeafternoon")
def stationdaytimeRouteAfternoon():
    print("Station id by daytime rentals Afternoon")
    #We need lat lon and predictions (and daytime later)
    resultlist=prediection2Select.toJSON().collect()
    #print(resultlist)
    json_results = json.dumps(resultlist)
    return json_results 

@app.route("/getstationdaytimeevening")
def stationdaytimeRouteEvening():
    print("Station id by daytime rentals Evening")
    #We need lat lon and predictions (and daytime later)
    resultlist=prediection3Select.toJSON().collect()
    #print(resultlist)
    json_results = json.dumps(resultlist)
    return json_results  

@app.route("/getstationdaytimenight")
def stationdaytimeRouteNight():
    print("Station id by daytime rentals Night")
    #We need lat lon and predictions (and daytime later)
    resultlist=prediection4Select.toJSON().collect()
    #print(resultlist)
    json_results = json.dumps(resultlist)
    return json_results     
    
@app.route("/traindata")
def trainRoute():
    print("Training data")
    # Read Data from database
    #mongoClient = MongoClient('mongodb://admin:admin@mongodb/sampledb')
    # run learning on apache spark  
    return null

####### Allowing access-control
#@app.after_request
#def after_request(response):
#  response.headers.add('Access-Control-Allow-Origin', '*')
#  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#  return response


    
    
print("HTTP Server started")
app.run(host='0.0.0.0', port=8080)


#spark.stop()
