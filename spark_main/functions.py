import random

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


def add_validation_column(temp_sdf):
    def validation(x):
        res = []
        for row in x:
            v = True
            if (len(row) != 24 or  # unusual size
                    (row >= 0).sum() != 24 or  # number of valid elements = 24
                    # sum(n >= 0 for n in row) != 24 or
                    # equal or more than 3 zero elements
                    np.count_nonzero(row == 0) >= 3 or
                    sum(n < 0 for n in row) > 0):  # not have negative element
                v = False
            res.append(v)
        return pd.Series(res)

    validation_UDF = pandas_udf(validation, returnType=BooleanType())
    temp = temp_sdf.withColumn("V", validation_UDF(col("power")))
    return temp


# add "N"ormal consumption ("N"onmalicious) column
def add_normal_column(temp_sdf):
    N = True
    temp = temp_sdf.withColumn("N", f.lit(N))
    return temp


# generate unique id
def generate_unique_id(temp_sdf):
    temp = temp_sdf
    temp = temp.withColumn("uid", f.concat(col("id"), f.lit("-"), col("#")).alias("uid"))
    return temp


# Generate malicious samples
def h1(x):
    MAX = 0.8
    MIN = 0.1
    alpha = random.uniform(MIN, MAX)
    temp = np.array(x)
    return (temp * alpha).tolist()


def h2(x):
    MIN_OFF = 4  # hour
    DURATION = random.randint(MIN_OFF, 23)
    START = random.randint(0, 23 - DURATION) if DURATION != 23 else 0
    END = START + DURATION
    temp = []
    for i in range(len(x)):
        if i < START or i >= END:
            temp.append(x[i])
        else:
            temp.append(0.0)
    return temp


def h3(x):
    MAX = 0.8
    MIN = 0.1
    temp = []
    for i in range(len(x)):
        temp.append(x[i] * random.uniform(MIN, MAX))
    return temp


def h4(x):
    MAX = 0.8
    MIN = 0.1
    mean = np.mean(x)
    temp = []
    for i in range(len(x)):
        temp.append(mean * random.uniform(MIN, MAX))
    return temp


def h5(x):
    mean = np.mean(x)
    temp = []
    for i in range(len(x)):
        temp.append(mean)
    return temp


def h6(x):
    temp = np.array(x)
    # temp=temp[::-1]
    temp = np.flipud(temp)
    return temp.tolist()


# add malicious samples
def create_malicious_df(temp_sdf):
    def random_attack_assigner(x):
        NUMBER_OF_MALICIOUS_GENERATOR = 6
        res = []
        for row in x:
            rand = random.randint(1, NUMBER_OF_MALICIOUS_GENERATOR)
            if rand == 1:
                temp = (h1(row))
            elif rand == 2:
                temp = (h2(row))
            elif rand == 3:
                temp = (h3(row))
            elif rand == 4:
                temp = (h4(row))
            elif rand == 5:
                temp = (h5(row))
            elif rand == 6:
                temp = (h6(row))
            else:
                temp = None
            res.append(temp)
        return pd.Series(res)

    cols = temp_sdf.columns
    random_attack_assigner_UDF = pandas_udf(random_attack_assigner, returnType=ArrayType(FloatType()))
    N = False
    temp_sdf = temp_sdf.withColumn("N", f.lit(N))  # malicious sample
    # change '#' column number to negative
    temp_sdf = temp_sdf.withColumn("#", col("#") * -1)
    temp_sdf = temp_sdf.withColumn("power", random_attack_assigner_UDF(col("power")))
    temp_sdf = add_statistics_column(temp_sdf)  # for update statistics
    return temp_sdf.select(cols)  # to reorder columns
