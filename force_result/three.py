import random

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from persiantools.jdatetime import JalaliDateTime


def load_data_frame():
    temp_df = pd.read_csv("my_data/all_data.csv", date_parser=["datetime"])
    temp_df.date = pd.to_datetime(temp_df.date)
    temp_df = temp_df[["id", "date", "usage"]]
    temp_df = temp_df.set_index("date")
    temp_df = temp_df.groupby(["id"]).apply(calculate_usage)
    temp_df = temp_df.reset_index(level="id", drop=False)
    return temp_df


def select_one_user(temp_df: pd.DataFrame, user_id: int):
    return temp_df[temp_df["id"] == user_id]


def select_random_user(temp_df: pd.DataFrame):
    return temp_df[temp_df["id"] == temp_df["id"][random.randint(0, temp_df["id"].shape[0])]]


def data_frame_agg(temp_df: pd.DataFrame, agg_type: str = "1H"):
    temp_df = temp_df.groupby("id").resample(agg_type).agg({"usage": "sum"})
    temp_df = temp_df.reset_index(level="id", drop=False)
    return temp_df


def data_frame_std_agg(temp_df: pd.DataFrame, agg_type: str = "1H"):
    temp_df = temp_df.groupby("id").resample(agg_type).agg({"semi_var": "sum"})
    temp_df = temp_df.reset_index(level="id", drop=False)
    return temp_df


def data_frame_all_agg(temp_df: pd.DataFrame, agg_type: str = "1H"):
    temp_dict = {}
    for col in temp_df.columns:
        temp_dict[col] = "sum"
    try:
        temp_dict.pop("id")
    except:
        pass
    temp_df = temp_df.groupby("id").resample(agg_type).agg(temp_dict)
    temp_df = temp_df.reset_index(level="id", drop=False)
    return temp_df


def filter_date(temp_df: pd.DataFrame, start: pd.Timestamp = None, end: pd.Timestamp = None):
    # temp_date = JalaliDateTime(1399, 5, 4)
    # start = pd.Timestamp(temp_date.to_gregorian())

    if start is not None:
        temp_df = temp_df[temp_df.index > start]
    if end is not None:
        temp_df = temp_df[temp_df.index < end]
    return temp_df


def set_detection_parameters_optimizer(resample, windows_size, std, new_m, new_n):
    global resample_value
    global moving_avg_windows_size
    global std_coe
    global m
    global n
    resample_value = "{}H".format(12 + resample)
    moving_avg_windows_size = int(10 + windows_size)
    std_coe = 1 + std
    m = int(6 + new_m)
    n = int(m * new_n)


def set_detection_parameters(resample: str, windows_size: int, std: float, new_m: int, new_n: int):
    global resample_value
    global moving_avg_windows_size
    global std_coe
    global m
    global n
    resample_value = resample
    moving_avg_windows_size = windows_size
    std_coe = std
    m = new_m
    n = new_n


def calculate_usage(temp_df):
    temp_df = temp_df.resample("1H").agg({"usage": "max"})
    temp_df["usage"] = temp_df["usage"] - temp_df["usage"].shift(1)
    # temp_df = temp_df.dropna()
    return temp_df


def usage_mean_below(temp_df: pd.DataFrame, threshold: int, threshold_range: str = "1D"):
    threshold_df = temp_df.groupby("id").resample(threshold_range).agg({"usage": "sum"})
    threshold_df = threshold_df.reset_index(level="id", drop=False)
    threshold_df = threshold_df.groupby("id").agg({"usage": "mean"})
    threshold_df = threshold_df[threshold_df.usage < threshold]
    threshold_df = threshold_df.reset_index(level="id", drop=False)
    return temp_df[temp_df.id.isin(threshold_df.id)]


def usage_mean_above(temp_df: pd.DataFrame, threshold: int, threshold_range: str = "1D"):
    threshold_df = temp_df.groupby("id").resample(threshold_range).agg({"usage": "sum"})
    threshold_df = threshold_df.reset_index(level="id", drop=False)
    threshold_df = threshold_df.groupby("id").agg({"usage": "mean"})
    threshold_df = threshold_df[threshold_df.usage > threshold]
    threshold_df = threshold_df.reset_index(level="id", drop=False)
    return temp_df[temp_df.id.isin(threshold_df.id)]


def detect(temp_df: pd.DataFrame):
    temp_df = temp_df.groupby(["id"]).apply(calculate_bands)
    temp_df = temp_df.reset_index(level="id", drop=False)
    return temp_df


def calculate_bands(temp_df: pd.DataFrame):
    temp_df = temp_df.resample(resample_value).agg({"usage": "sum"})
    temp_df = temp_df.dropna()
    temp_df["avg_value"] = temp_df.usage.rolling(moving_avg_windows_size, min_periods=1).mean()
    temp_df["std_value"] = temp_df.usage.rolling(moving_avg_windows_size, min_periods=1).std()
    temp_df["upper_band"] = temp_df.avg_value + temp_df.std_value * std_coe
    temp_df["lower_band"] = temp_df.avg_value - temp_df.std_value * std_coe
    return temp_df


def day_night_usage_filter(temp_df: pd.DataFrame, day_mean_above: float = None, night_mean_above: float = None,
                           day_mean_below: float = None, night_mean_below: float = None):
    def calculate_day_night_mean(inner_df: pd.DataFrame):
        day_mean = inner_df[inner_df["day"]]["usage"].mean()
        night_mean = inner_df[~inner_df["day"]]["usage"].mean()
        return pd.Series([day_mean, night_mean])

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
    return temp_df[["id", "usage"]]


def get_std_df(temp_df: pd.DataFrame):
    def calculate_mean(inner_df: pd.DataFrame):
        inner_df = inner_df.resample("1D").agg({"usage": "sum"})
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=moving_avg_windows_size)
        inner_df["avg_value"] = inner_df.usage.rolling(window=indexer, min_periods=1).mean()
        inner_df["std_value"] = inner_df.usage.rolling(window=indexer, min_periods=1).std()
        inner_df["semi_var"] = inner_df["std_value"] / inner_df["avg_value"]
        return inner_df[["semi_var"]]

    temp_df = temp_df.groupby(["id"]).apply(calculate_mean)
    temp_df = temp_df.reset_index(level="id", drop=False)
    return temp_df


def usage_mean_above_input_percent(temp_df: pd.DataFrame, correct_above: float, resample_type: str,
                                   daily_usage_above: int):
    def above_threshold(inner_df: pd.DataFrame):
        val = inner_df["high"].sum() / inner_df["high"].count()
        if val > correct_above:
            inner_df["good"] = True
        else:
            inner_df["good"] = False
        return inner_df

    daily_temp_df = temp_df.groupby("id").resample(resample_type).agg({"usage": "sum"})
    daily_temp_df = daily_temp_df.reset_index("id", drop=False)
    daily_temp_df["high"] = daily_temp_df.usage > daily_usage_above
    daily_temp_df = daily_temp_df.groupby("id").apply(above_threshold)
    daily_temp_df = daily_temp_df[daily_temp_df["good"]]
    return temp_df[temp_df.id.isin(daily_temp_df.id.unique())]


def usage_mean_below_input_percent(temp_df: pd.DataFrame, correct_above: float, resample_type: str,
                                   daily_usage_below: int):
    def above_threshold(inner_df: pd.DataFrame):
        val = inner_df["high"].sum() / inner_df["high"].count()
        if val > correct_above:
            inner_df["good"] = True
        else:
            inner_df["good"] = False
        return inner_df

    daily_temp_df = temp_df.groupby("id").resample(resample_type).agg({"usage": "sum"})
    daily_temp_df = daily_temp_df.reset_index("id", drop=False)
    daily_temp_df["high"] = daily_temp_df.usage < daily_usage_below
    daily_temp_df = daily_temp_df.groupby("id").apply(above_threshold)
    daily_temp_df = daily_temp_df[daily_temp_df["good"]]
    return temp_df[temp_df.id.isin(daily_temp_df.id.unique())]


def plot(temp_df: pd.DataFrame, user_id: int, fig_name: str, columns: list):
    temp_df = select_one_user(temp_df, user_id)
    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 5 * len(columns)))
    if len(columns) > 1:
        for i, col in enumerate(columns):
            axes[i].plot(temp_df.index.map(JalaliDateTime).map(str), temp_df[col])
            locator = mdates.AutoDateLocator(minticks=10, maxticks=15)
            axes[i].xaxis.set_major_locator(locator)

            for label in axes[i].get_xticklabels():
                label.set_rotation(20)
                label.set_horizontalalignment('right')
    else:
        axes.plot(temp_df.index.map(JalaliDateTime).map(str), temp_df[columns[0]])
        locator = mdates.AutoDateLocator(minticks=10, maxticks=15)
        axes.xaxis.set_major_locator(locator)

        for label in axes.get_xticklabels():
            label.set_rotation(20)
            label.set_horizontalalignment('right')

    fig.tight_layout()
    plt.savefig(fig_name)


def plot_detection(temp_df: pd.DataFrame, mining_prediction: pd.DataFrame, theft_prediction: pd.DataFrame,
                   user_id: int, fig_name: str):
    temp_df = select_one_user(temp_df, user_id)
    fig, axe = plt.subplots(1, 1, figsize=(10, 5))
    indexes = temp_df.index.map(JalaliDateTime).map(str)
    axe.plot(indexes, temp_df["usage"], 'b')
    temp_mining = mining_prediction[mining_prediction.id == user_id]
    temp_theft = theft_prediction[theft_prediction.id == user_id]
    if temp_mining.shape[0] > 0:
        axe.plot(temp_mining.index.map(JalaliDateTime).map(str), temp_mining["usage"], "r", marker="*",
                 linestyle=None, linewidth=0, markersize=11, label="mining")
    if temp_theft.shape[0] > 0:
        axe.plot(temp_theft.index.map(JalaliDateTime).map(str), temp_theft["usage"], "yellow", marker="*",
                 linestyle=None, linewidth=0, markersize=11, label="theft")
    locator = mdates.AutoDateLocator(minticks=10, maxticks=15)
    axe.xaxis.set_major_locator(locator)
    axe.legend()

    for label in axe.get_xticklabels():
        label.set_rotation(20)
        label.set_horizontalalignment('right')
    fig.tight_layout()
    plt.savefig(fig_name)
    plt.close()


def calculate_bands_new_method(temp_df: pd.DataFrame):
    temp_df = temp_df.resample(resample_value).agg({"usage": "sum"})
    usage = temp_df.usage
    temp_df["mining"] = False
    temp_df["theft"] = False
    temp_df["to_use"] = True
    start = moving_avg_windows_size
    while start < usage.shape[0]:
        temp_data = usage[start]
        temp_window = usage.iloc[start - moving_avg_windows_size:][
                          temp_df.iloc[start - moving_avg_windows_size:]["to_use"]][
                      0:moving_avg_windows_size].copy()

        temp_avg = temp_window.mean()
        temp_std = temp_window.std()
        temp_upper = temp_avg + (temp_std * std_coe)
        temp_lower = temp_avg - (temp_std * std_coe)

        if temp_data > temp_upper:
            step = 1
            while True:
                m_window = usage[start + step:start + step + m]
                m_window_detect = (m_window > temp_upper) & (m_window > 20)
                m_window_ok = m_window_detect.sum() < n
                if m_window_ok:
                    break
                step += 1
            if step > 1:
                temp_df["mining"][start + 1:start + step + m] = (usage[start + 1:start + step + m] > temp_upper)
                temp_df["to_use"][start + 1:start + step + m] = (usage[start + 1:start + step + m] < temp_upper)
                start = start + moving_avg_windows_size + step

        if temp_data < temp_lower:
            step = 1
            while True:
                m_window = usage[start + step:start + step + m]
                m_window_detect = (m_window < temp_lower)
                m_window_ok = m_window_detect.sum() < n
                if m_window_ok:
                    break
                step += 1
            if step > 1:
                temp_df["theft"][start + 1:start + step + m] = (usage[start + 1:start + step + m] < temp_lower)
                temp_df["to_use"][start + 1:start + step + m] = (usage[start + 1:start + step + m] > temp_lower)
                start = start + moving_avg_windows_size + step

        start += 1
    return temp_df


def plot_new_detection(temp_df: pd.DataFrame, temp_user_id: int, fig_name: str):
    temp_df = select_one_user(temp_df, temp_user_id)
    fig, axe = plt.subplots(1, 1, figsize=(10, 5))
    indexes = temp_df.index.map(JalaliDateTime).map(str)
    axe.plot(indexes, temp_df["usage"], 'black', label="usage")
    temp_df["mining_value"] = np.inf
    temp_df["mining_value"] = temp_df["usage"][temp_df["mining"]]
    temp_df["theft_value"] = np.inf
    temp_df["theft_value"] = temp_df["usage"][temp_df["theft"]]
    if temp_df[temp_df["mining"]].count()[0] > 0:
        axe.plot(indexes, temp_df["mining_value"], 'yellow', label="mining")
    if temp_df[temp_df["theft"]].count()[0] > 0:
        axe.plot(indexes, temp_df["theft_value"], 'r', label="theft")
    locator = mdates.AutoDateLocator(minticks=10, maxticks=15)
    axe.xaxis.set_major_locator(locator)
    axe.legend()

    for label in axe.get_xticklabels():
        label.set_rotation(20)
        label.set_horizontalalignment('right')
    fig.tight_layout()
    plt.savefig(fig_name)
    plt.close()


def new_detect(temp_df: pd.DataFrame):
    temp_df = temp_df.groupby(["id"]).apply(calculate_bands_new_method)
    temp_df = temp_df.reset_index(level="id", drop=False)
    return temp_df


def calculate_accuracy(a):
    set_detection_parameters_optimizer(a[0], a[1], a[2], a[3], a[4])
    correct_count = 0
    for user_id in good_users:
        temp_user = select_one_user(main_df, user_id)
        user_label = labels[labels.img == user_id].all()
        temp_detection = new_detect(temp_user)
        is_mining = temp_detection.mining.sum() > 0
        is_theft = temp_detection.theft.sum() > 0
        if user_label.normal:
            if not is_mining or not is_theft:
                correct_count += 1
        elif user_label.mining:
            if is_mining:
                correct_count += 1
        elif user_label.theft:
            if is_theft:
                correct_count += 1
        elif is_theft or is_mining:
            correct_count += 1
    return 1 - (correct_count / len(good_users))


resample_value = "{}H".format(24)
moving_avg_windows_size = 40
std_coe = 2.5
m = 20
n = 15

labels = pd.read_csv("my_data/labels.csv")
good_users = random.sample(labels[labels.unknown != 1].img.tolist(), 10)
# good_users = labels[labels.unknown != 1].img.tolist()
main_df = load_data_frame()
