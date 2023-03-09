###K-MEANS###
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
import matplotlib.pyplot as plt


spark = SparkSession.builder.master("local[*]").appName("Clustering").getOrCreate()
print("########################")

#1. ucitavanje skupa podataka
data = spark.read.format("csv").option("inferSchema", "true").option("header", "true").load('iris.csv')
data.show()
#2. analiza skupa podataka - na osnovu izgleda cveta odredjuje se koja je vrsta iris cveta
# data.describe().show()
print("###Opisni podaci skupa###")
print(data.count(),"," ,len(data.columns))
data.summary().show() #detaljnija describe
print(data.printSchema()) 
data.summary("count").show()

#3. sredivanje podataka
    #nedostajuce vrednosti
print("###Provera nedostajucih vrednosti###")
data.select([count(when(isnan(c), c)).alias(c) for c in data.columns]).show()

    #normalizacija
columns_names = ["sepal_length","sepal_width","petal_length","petal_width"]
for column in columns_names:
    assembler = VectorAssembler(inputCols=[column],outputCol=column+"_vectorized") #potrebno je da napravimo kolonu da bude vektor
    data = assembler.transform(data)

    mmScaler = MinMaxScaler(inputCol=column+"_vectorized", outputCol=column+"_normalized")
    model = mmScaler.fit(data)
    data = model.transform(data)

    unlist = udf(lambda x: float(list(x)[0]), DoubleType()) #udf za konvertovanje kolone iz vektora u double
    data = data.withColumn(column+"_last", unlist(column+"_normalized"))

print("###Prikaz normalizovanog skupa podataka###")
data2 = data.select("sepal_length_last", "sepal_width_last", "petal_length_last", "petal_width_last", "variety")
data2.show(5)

    #autlajeri
columns_names = ["sepal_length_last", "sepal_width_last", "petal_length_last", "petal_width_last"]
for column in columns_names:
    pdf = data2.select(F.col(column)).toPandas()
    pdf.plot.box(title="Boxplot of Values for " + column)
    plt.show()

#sepal_width_last ima autlajere - uklanjanje
summary_stats = data2.select("sepal_width_last").summary("25%", "75%")

q1 = float(summary_stats.filter(col("summary") == "25%").select("sepal_width_last").first()[0])
q3 = float(summary_stats.filter(col("summary") == "75%").select("sepal_width_last").first()[0])

iqr = q3 - q1

lower_range = q1 - 1.5 * iqr
upper_range = q3 + 1.5 * iqr

df_no_outliers = data2.filter(col("sepal_width_last").between(lower_range, upper_range))
print("###No outliers###")
df_no_outliers.show(5)
print(df_no_outliers.count())

pdf = df_no_outliers.select(F.col("sepal_width_last")).toPandas()
pdf.plot.box(title="Boxplot of Values for " + "sepal_width_last")
plt.show()

# 4. pravljenje modela
    # grupisanje
input_columns = ["sepal_length_last","sepal_width_last","petal_length_last","petal_width_last"]
vector_assembler = VectorAssembler(inputCols= input_columns, outputCol="features") #sve kolone koje ulaze u analizu stavljam u jednu kolonu 

data_transformed = vector_assembler.transform(df_no_outliers)
print("###Transformisani podaci###")
data_transformed.show(5)

    # izrada modela
kmeans = KMeans(featuresCol="features").setK(3).setSeed(3)
model = kmeans.fit(data_transformed)  
print(model)
print(model.transform(data_transformed).groupBy("prediction").count().show())

# 6. provera modela

predictions = model.transform(data_transformed)

#procena modela
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

predictions.show(5)

print(predictions.groupBy("variety", "prediction").count().show())

