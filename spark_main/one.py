from pyspark.sql import SparkSession

from spark_main.functions import *

spark = SparkSession.builder.appName("anomaly_detection").getOrCreate()

# define schema
schema = StructType([StructField("#", IntegerType()),
                     StructField("date", TimestampType()),
                     StructField("id", StringType()),
                     StructField("power", StringType())])

# read data
sdf = spark.read.format('csv').options(header='true',
                                       inferSchema=True,
                                       schema=schema).load("my_data/spark_readable.csv")

sdf = rename_data_frame(sdf)
sdf = string_power_to_array(sdf)
sdf = add_statistics_column(sdf)
train_percent = 0.4

users_data_count = sdf.groupby("id").agg(f.count(sdf.date).alias("count"),
                                         f.min(sdf.date).alias("min"),
                                         f.max(sdf.date).alias("max"),
                                         f.datediff(f.max(sdf.date), f.min(sdf.date)).alias("date_diff"))
users_data_count = users_data_count.withColumn("train_count", f.floor(f.col("count") * train_percent))
users_data_count = users_data_count.withColumn("test_count", f.ceil(f.col("count") * (1 - train_percent)))
users_data_count = users_data_count.withColumn("split_date", f.expr("date_add(min, train_count)"))
# users_data_count = users_data_count.withColumn("split_date", f.date_add("min", f.col("train_count")))
# date_add_udf = f.udf(lambda date, days: f.date_add(date, days), TimestampType())
# users_data_count = users_data_count.withColumn("split_date", date_add_udf(f.col("min"), f.col("train_count")))

users_split_data = users_data_count.select(["id", "split_date"])

sdf = sdf.join(users_split_data, "id", "inner")
train_sdf = sdf.filter(sdf.date < sdf.split_date).select(["#", "date", "id", "power", "statistics"])
test_sdf = sdf.filter(sdf.date >= sdf.split_date).select(["#", "date", "id", "power", "statistics"])

train_sdf.toPandas().to_pickle("my_data/train_data.pkl")
test_sdf.toPandas().to_pickle("my_data/test_data.pkl")
