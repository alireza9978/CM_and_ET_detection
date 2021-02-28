from pyspark.sql import SparkSession

from spark_main.functions import *

spark = SparkSession.builder.appName("anomaly_detection").getOrCreate()

# load sdf
print("-------------------- loading data set!")
df = pd.read_pickle("my_data/train_data.pkl")
sdf = spark.createDataFrame(df)

sdf = add_validation_column(sdf)
sdf = add_normal_column(sdf)
sdf = filter_data_set(sdf, from_date="BEGIN", to_date="END", temp_id="*", validation="True")  # 2016-07-01: 75%
sdf = generate_unique_id(sdf)

# generate malicious data
sdf_malicious = create_malicious_df(sdf)
sdf_mix = sdf.union(sdf_malicious)
df_mix = sdf_mix.toPandas()

df_mix.to_pickle("my_data/train_mix.pkl")


