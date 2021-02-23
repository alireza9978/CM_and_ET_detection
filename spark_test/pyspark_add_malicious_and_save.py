from pyspark.sql import SparkSession

from spark_test.functions import *

spark = SparkSession.builder.appName("anomaly_detection").getOrCreate()

# define schema
schema = StructType([
    StructField("#", IntegerType()),
    StructField("date", TimestampType()),
    StructField("id", StringType()),
    StructField("power", StringType())])

# read data
sdf = spark.read.format('csv').options(header='true', inferSchema=True, schema=schema).load(
    "my_data/spark_readable.csv")

sdf = rename_data_frame(sdf)
sdf = string_power_to_array(sdf)
sdf = add_statistics_column(sdf)
sdf = add_validation_column(sdf)
sdf = add_normal_column(sdf)
sdf = filter_data_set(sdf, from_date="BEGIN", to_date="END", temp_id="*", validation="True")  # 2016-07-01: 75%
sdf = generate_unique_id(sdf)

# generate malicious data
sdf_malicious = create_malicious_df(sdf)
sdf_mix = sdf.union(sdf_malicious)
df_mix = sdf_mix.toPandas()

df_mix.to_pickle("my_data/iran*.pkl")

