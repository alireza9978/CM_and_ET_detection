import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from persiantools.jdatetime import JalaliDateTime

# load date set and convert date
data_frame = pd.read_csv("my_data/all_data.csv", date_parser=["datetime"])
data_frame.date = pd.to_datetime(data_frame.date)
data_frame = data_frame[["id", "date", "usage"]]
data_frame = data_frame.set_index("date")
data_frame["usage"] = data_frame["usage"] - data_frame["usage"].shift(1)


def plot_user(user_id, index: pd.Timestamp = None, hour=True, six_hour=True, half_day=True, day=True, week=True):
    # select a random user from all users
    user_df = data_frame[data_frame["id"] == user_id]

    # nazdik tarin index be index vorodi ro to list peida mikone
    def nearest(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))

    data_frame_plot_range = [240, 100, 100, 15, 15]
    data_frames = []
    if hour:
        data_frames.append((user_df, data_frame_plot_range[0], "hourly"))
    if six_hour:
        data_frames.append((user_df.resample("6H").agg({"usage": "sum"}), data_frame_plot_range[1], "six_hourly"))
    if half_day:
        data_frames.append((user_df.resample("12H").agg({"usage": "sum"}), data_frame_plot_range[2], "half_day"))
    if day:
        data_frames.append((user_df.resample("1D").agg({"usage": "sum"}), data_frame_plot_range[3], "daily"))
    if week:
        data_frames.append((user_df.resample("7D").agg({"usage": "sum"}), data_frame_plot_range[4], "weekly"))

    if index is not None:
        middles = []
        for temp_df, plot_range, name in data_frames:
            middles.append((temp_df.index.get_loc(nearest(temp_df.index, index)), len(temp_df.index), plot_range))

        end = []
        start = []
        for temp in middles:
            if temp[0] - temp[2] < 0:
                start.append(0)
            else:
                start.append(temp[0] - temp[2])
            if temp[0] + temp[2] > temp[1]:
                end.append(temp[1])
            else:
                end.append(temp[0] + temp[2])
        for i in range(len(data_frames)):
            data_frames[i] = (data_frames[i][0][start[i]:end[i]], data_frames[i][1], data_frames[i][2])

    fig, axes = plt.subplots(nrows=len(data_frames), ncols=1, figsize=(10, 4 * len(data_frames)))
    for i, (temp_df, plot_range, name) in enumerate(data_frames):
        axes[i].plot(temp_df.index.map(JalaliDateTime).map(str), temp_df["usage"])
        locator = mdates.AutoDateLocator(minticks=10, maxticks=15)
        axes[i].xaxis.set_major_locator(locator)

        for label in axes[i].get_xticklabels():
            label.set_rotation(20)
            label.set_horizontalalignment('right')
        axes[i].set_title(name)
    fig.tight_layout()

    target_path = "my_figures/force_result/{}".format(user_id)
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    if index is not None:
        plt.savefig(target_path + "/main_middle_{}.jpg".format(middles[0][0]))
    else:
        plt.savefig(target_path + "/all_data.jpg")
    plt.close()
