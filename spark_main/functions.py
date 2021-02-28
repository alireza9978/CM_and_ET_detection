import numpy as np
import pandas as pd
import pyspark.sql.functions as f
from pyspark.sql.functions import col, split, pandas_udf, udf
from pyspark.sql.types import *


@udf(returnType=ArrayType(FloatType()))
def generate_feature_udf(x):
    if x is not None:
        row = np.array(x, dtype=np.float)
        if row.shape[0] >= 1:
            return [np.nanmean(row), np.nanstd(row), np.nanmin(row), np.nanmax(row)]
        return []
    return []


# generate feature
def generate_feature(x):
    res = []
    for row in x:
        row = np.array(row)  # to numpy
        statistics = []
        min_val = np.nanmin(row)
        max_val = np.nanmax(row)
        mean_val = np.nanmean(row)
        std_val = np.nanstd(row)
        statistics.append(mean_val)
        statistics.append(std_val)
        statistics.append(min_val)
        statistics.append(max_val)
        res.append(statistics)
    return pd.Series(res)


def add_statistics_column(temp_sdf):
    temp = temp_sdf.withColumn("statistics", generate_feature_UDF(col("power")))
    return temp


# rename columns
def rename_data_frame(temp_sdf):
    names = ['#', 'date', 'id', 'power']
    for c, n in zip(temp_sdf.columns, names):
        temp_sdf = temp_sdf.withColumnRenamed(c, n)
    return temp_sdf


def string_power_to_array(temp_sdf):
    temp = temp_sdf.withColumn("power", f.regexp_replace(f.regexp_replace(f.col("power"), "\\[", ""), "\\]", "")
                               .alias("power"))
    temp = temp.withColumn("power", split(col("power"), ",\s*").cast(ArrayType(FloatType())).alias("power"))
    return temp


def filter_data_set(temp_sdf, from_date="BEGIN", to_date="END", temp_id="*", validation="*"):
    temp = temp_sdf
    if from_date != "BEGIN":
        temp = temp.filter(temp_sdf.date > from_date)  # filter date (from X)
    if to_date != "END":
        temp = temp.filter(temp_sdf.date < to_date)  # filter date (to Y)
    if temp_id != "*":
        temp = temp.filter(temp_sdf.id == temp_id)  # filter IDs
    if validation != "*":
        temp = temp.filter(temp_sdf.V == validation)  # filter validation
    return temp


generate_feature_UDF = pandas_udf(generate_feature, returnType=ArrayType(FloatType()))
