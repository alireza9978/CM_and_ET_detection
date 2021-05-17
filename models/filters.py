import random

import pandas as pd


# return data that belong to entered user
def select_one_user(temp_df: pd.DataFrame, user_id: int):
    return temp_df[temp_df["id"] == user_id]


# return data that belong to random user
def select_random_user(temp_df: pd.DataFrame):
    return temp_df[temp_df["id"] == temp_df["id"][random.randint(0, temp_df["id"].shape[0])]]


# change usage frequency by adding usage in each period
def data_frame_agg(temp_df: pd.DataFrame, agg_type: str = "1H"):
    temp_df = temp_df.set_index("date")
    temp_df = temp_df.groupby("id").resample(agg_type).agg({"usage": "sum"})
    temp_df = temp_df.reset_index(drop=False)
    return temp_df


# return data that are after start and before end
def filter_date(temp_df: pd.DataFrame, start: pd.Timestamp = None, end: pd.Timestamp = None):
    # from hejri shamsi
    # temp_date = JalaliDateTime(1399, 5, 4)
    # start = pd.Timestamp(temp_date.to_gregorian())

    # from normal date
    # start = pd.to_datetime(datetime(2020, 8, 2))

    temp_df = temp_df.set_index("date")
    if start is not None:
        temp_df = temp_df[temp_df.index > start]
    if end is not None:
        temp_df = temp_df[temp_df.index < end]
    temp_df = temp_df.reset_index(drop=False)
    return temp_df


# return data of users that their single usage is above input and this condition is true for more than input percent
def usage_mean_above_input_percent(temp_df: pd.DataFrame, correct_above: float, resample_type: str,
                                   daily_usage_above: int):
    # 0 >= correct_above >= 1
    def above_threshold(inner_df: pd.DataFrame):
        val = inner_df["high"].sum() / inner_df["high"].count()
        if val > correct_above:
            inner_df["good"] = True
        else:
            inner_df["good"] = False
        return inner_df

    daily_temp_df = temp_df.set_index("date")
    daily_temp_df = daily_temp_df.groupby("id").resample(resample_type).agg({"usage": "sum"})
    daily_temp_df = daily_temp_df.reset_index("id", drop=False)
    daily_temp_df["high"] = daily_temp_df.usage > daily_usage_above
    daily_temp_df = daily_temp_df.groupby("id").apply(above_threshold)
    daily_temp_df = daily_temp_df[daily_temp_df["good"]]
    return temp_df[temp_df.id.isin(daily_temp_df.id.unique())]


# return data of users that their single usage is below input and this condition is true for more than input percent
def usage_mean_below_input_percent(temp_df: pd.DataFrame, correct_above: float, resample_type: str,
                                   daily_usage_below: int):
    def above_threshold(inner_df: pd.DataFrame):
        val = inner_df["low"].sum() / inner_df["low"].count()
        if val > correct_above:
            inner_df["good"] = True
        else:
            inner_df["good"] = False
        return inner_df

    daily_temp_df = temp_df.set_index("date")
    daily_temp_df = daily_temp_df.groupby("id").resample(resample_type).agg({"usage": "sum"})
    daily_temp_df = daily_temp_df.reset_index("id", drop=False)
    daily_temp_df["low"] = daily_temp_df.usage < daily_usage_below
    daily_temp_df = daily_temp_df.groupby("id").apply(above_threshold)
    daily_temp_df = daily_temp_df[daily_temp_df["good"]]
    return temp_df[temp_df.id.isin(daily_temp_df.id.unique())]


# return data of users, that average of usage in night and day of a user is between inputs
def day_night_usage_filter(temp_df: pd.DataFrame, day_mean_above: float = None, night_mean_above: float = None,
                           day_mean_below: float = None, night_mean_below: float = None):
    def calculate_day_night_mean(inner_df: pd.DataFrame):
        day_mean = inner_df[inner_df["day"]]["usage"].mean()
        night_mean = inner_df[~inner_df["day"]]["usage"].mean()
        return pd.Series([day_mean, night_mean])

    temp_df = temp_df.set_index("date")
    temp_df["day"] = False
    temp_df["day"] = (temp_df.index.hour > 5) & (temp_df.index.hour < 18)
    day_night_mean_df = temp_df.groupby("id").apply(calculate_day_night_mean)
    day_night_mean_df = day_night_mean_df.reset_index(level="id", drop=False)

    if day_mean_above is not None:
        temp_df = temp_df[temp_df.id.isin(day_night_mean_df[(day_night_mean_df[0] > day_mean_above)]["id"])]
    if day_mean_below is not None:
        temp_df = temp_df[temp_df.id.isin(day_night_mean_df[(day_night_mean_df[0] < day_mean_below)]["id"])]
    if night_mean_above is not None:
        temp_df = temp_df[temp_df.id.isin(day_night_mean_df[(day_night_mean_df[1] > night_mean_above)]["id"])]
    if night_mean_below is not None:
        temp_df = temp_df[temp_df.id.isin(day_night_mean_df[(day_night_mean_df[1] < night_mean_below)]["id"])]
    temp_df = temp_df.reset_index(drop=False)
    return temp_df[["id", "date", "usage"]]


# return data of users, that average usage of a user is less than inputs
def usage_mean_below(temp_df: pd.DataFrame, threshold: float, resample_type: str = None):
    temp_df = temp_df.set_index("date")
    if resample_type is not None:
        threshold_df = temp_df.groupby("id").resample(resample_type).agg({"usage": "sum"})
    else:
        threshold_df = temp_df.groupby("id").agg({"usage": "sum"})
    threshold_df = threshold_df.reset_index(level="id", drop=False)
    threshold_df = threshold_df.groupby("id").agg({"usage": "mean"})
    threshold_df = threshold_df[threshold_df.usage < threshold]
    threshold_df = threshold_df.reset_index(level="id", drop=False)
    temp_df = temp_df[temp_df.id.isin(threshold_df.id)]
    return temp_df.reset_index(drop=False)


# return data of users, that average usage of a user is more than inputs
def usage_mean_above(temp_df: pd.DataFrame, threshold: float, resample_type: str = None):
    temp_df = temp_df.set_index("date")
    if resample_type is not None:
        threshold_df = temp_df.groupby("id").resample(resample_type).agg({"usage": "sum"})
    else:
        threshold_df = temp_df.groupby("id").agg({"usage": "sum"})
    threshold_df = threshold_df.reset_index(level="id", drop=False)
    threshold_df = threshold_df.groupby("id").agg({"usage": "mean"})
    threshold_df = threshold_df[threshold_df.usage > threshold]
    threshold_df = threshold_df.reset_index(level="id", drop=False)
    temp_df = temp_df[temp_df.id.isin(threshold_df.id)]
    return temp_df.reset_index(drop=False)
