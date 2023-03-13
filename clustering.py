###K-MEANS###
import pyspark.sql.functions as F
from pyspark.sql.functions import isnan, when, count, col, udf
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType

from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.clustering import KMeans, BisectingKMeans, BisectingKMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA as PCAml

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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
    # plt.show()

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
# plt.show()

# 4. pravljenje modela
    # grupisanje
input_columns = ["sepal_length_last","sepal_width_last","petal_length_last","petal_width_last"]
vector_assembler = VectorAssembler(inputCols= input_columns, outputCol="features") #sve kolone koje ulaze u analizu stavljam u jednu kolonu 

data_transformed = vector_assembler.transform(df_no_outliers)
print("###Transformisani podaci###")
data_transformed.show(5)

    # izrada modela
print("###Izrada K-menas modela###")
kmeans = KMeans(featuresCol="features").setK(3).setSeed(3)
model = kmeans.fit(data_transformed)  
print(model.transform(data_transformed).groupBy("prediction").count().show())

# 6. provera modela

predictions = model.transform(data_transformed)

#procena modela
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

predictions.show(5)

print(predictions.groupBy("variety", "prediction").count().show())

# Hijerarhijsko klasterovanje
print("###Izrada modela za hijerarhijsko kalsterovanje###")
hierarchical_clustering = BisectingKMeans(featuresCol="features").setK(3).setSeed(3)
model = hierarchical_clustering.fit(data_transformed)
print(model.transform(data_transformed).groupBy("prediction").count().show())
predictions2 = model.transform(data_transformed)

# provera
silhouette = evaluator.evaluate(predictions2)
print("Silhouette with squared euclidean distance = " + str(silhouette))

predictions2.show(5)

print(predictions2.groupBy("variety", "prediction").count().show())

#vizuelizacija sa PCA
pca = PCAml(k=2, inputCol="features", outputCol="pca")
pca_model = pca.fit(data_transformed)
pca_transformed = pca_model.transform(data_transformed)


X_pca = pca_transformed.rdd.map(lambda row: row.pca).collect()
X_pca = np.array(X_pca)

print("###Ovo je sada za k-means###")
cluster_assignment = np.array(predictions.rdd.map(lambda row: row.prediction).collect()).reshape(-1,1)

pca_data = np.hstack((X_pca,cluster_assignment))

pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal","cluster_assignment"))
sns.FacetGrid(pca_df,hue="cluster_assignment", height=6).map(plt.scatter, '1st_principal', '2nd_principal' ).add_legend()
plt.title("K-means")
plt.show()


print("###Ovo je sada hierarhijsko")
cluster_assignment = np.array(predictions2.rdd.map(lambda row: row.prediction).collect()).reshape(-1,1)

pca_data = np.hstack((X_pca,cluster_assignment))

pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal","cluster_assignment"))
sns.FacetGrid(pca_df,hue="cluster_assignment", height=6).map(plt.scatter, '1st_principal', '2nd_principal' ).add_legend()
plt.title("Hijerarhijsko klasterovanje###")
plt.show()

