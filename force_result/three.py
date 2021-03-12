import random

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from persiantools.jdatetime import JalaliDateTime


def load_data_frame():
    temp_df = pd.read_csv("my_data/good_data.csv", date_parser=["datetime"])
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


def set_detection_parameters(resample: str, windows_size: int, std: float):
    global resample_value
    global moving_avg_windows_size
    global std_coe
    resample_value = resample
    moving_avg_windows_size = windows_size
    std_coe = std


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
        # inner_df["avg_value"] = inner_df["avg_value"].replace(0, 1)
        inner_df["semi_var"] = inner_df["std_value"] / inner_df["avg_value"]
        # inner_df["semi_var"] = inner_df["semi_var"].where(inner_df["std_value"] > 0.1, inner_df["semi_var"].max())
        return inner_df[["semi_var"]]

    temp_df = temp_df.groupby(["id"]).apply(calculate_mean)
    temp_df = temp_df.reset_index(level="id", drop=False)
    return temp_df


def usage_mean_above_input_percent(temp_df: pd.DataFrame, correct_above: float, daily_usage_above: int):
    def above_threshold(inner_df: pd.DataFrame):
        val = inner_df["high"].sum() / inner_df["high"].count()
        if val > correct_above:
            inner_df["good"] = True
        else:
            inner_df["good"] = False
        return inner_df

    daily_temp_df = temp_df.groupby("id").resample("1D").agg({"usage": "sum"})
    daily_temp_df = daily_temp_df.reset_index("id", drop=False)
    daily_temp_df["high"] = daily_temp_df.usage > daily_usage_above
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


def test():
    temp_df = load_data_frame()
    temp_df = select_one_user(temp_df, 1502940805227)
    # temp_df = usage_mean_above(temp_df, 50, "1D")
    # temp_df = usage_mean_below(temp_df, 100, "1D")
    temp_df = data_frame_agg(temp_df, "1D")
    temp_df = filter_date(temp_df, pd.Timestamp(JalaliDateTime(1399, 9, 1).to_gregorian()),
                          pd.Timestamp(JalaliDateTime(1399, 9, 17).to_gregorian()))
    # temp_df = usage_mean_above_input_percent(temp_df, 0.8, 10)
    # detection
    # set_detection_parameters("7D", 40, 3.5)
    # temp_df = detect(temp_df)
    # mining_prediction = temp_df[temp_df.usage > temp_df.upper_band]
    # theft_prediction = temp_df[temp_df.usage < temp_df.lower_band]
    # print(mining_prediction)
    # print(theft_prediction)

    users = temp_df.id.unique()
    print(len(users))
    plot(temp_df, users[0], str(users[0]) + ".jpg", ["usage"])
    # plot_detection(temp_df, mining_prediction, theft_prediction, users[0], str(users[0]) + ".jpg")


resample_value = "{}H".format(24)
moving_avg_windows_size = 40
std_coe = 2.5
test()
