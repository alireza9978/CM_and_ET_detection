import os

import matplotlib.pyplot as plt
import pandas as pd

# load date set and convert date
data_frame = pd.read_csv("my_data/all_data.csv", date_parser=["datetime"])
data_frame.date = pd.to_datetime(data_frame.date)
data_frame = data_frame[["id", "date", "usage"]]


def plot_user(user_id, index: pd.Timestamp = None, hour=True, six_hour=True, half_day=True, day=True, week=True):
    # select a random user from all users
    user_df = data_frame[data_frame["id"] == user_id]

    # nazdik tarin index be index vorodi ro to list peida mikone
    def nearest(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))

    # ekhtelaf masraf ha ro hesab mikonim aval to scale haie mokhtalef be darsad
    # user_df["difference_hour"] = 100 * (user_df["usage"] - user_df["usage"].shift(1)) / user_df["usage"]
    user_df = user_df.set_index("date")
    user_df["usage"] = user_df["usage"] - user_df["usage"].shift(1)
    data_frame_plot_range = [240, 100, 100, 15, 15]
    data_frames = []
    if hour:
        data_frames.append((user_df, data_frame_plot_range[0]))
    if six_hour:
        data_frames.append((user_df.resample("6H").agg({"usage": "sum"}), data_frame_plot_range[1]))
    if half_day:
        data_frames.append((user_df.resample("12H").agg({"usage": "sum"}), data_frame_plot_range[2]))
    if day:
        data_frames.append((user_df.resample("1D").agg({"usage": "sum"}), data_frame_plot_range[3]))
    if week:
        data_frames.append((user_df.resample("7D").agg({"usage": "sum"}), data_frame_plot_range[4]))

    if index is not None:
        middles = []
        for temp_df, plot_range in data_frames:
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
            data_frames[i] = (data_frames[i][0][start[i]:end[i]], data_frames[i][1])

    fig, axes = plt.subplots(nrows=len(data_frames), ncols=1)
    fig.tight_layout()
    for i, (temp_df, plot_range) in enumerate(data_frames):
        number = (len(data_frames) * 100) + 11 + i
        plt.subplot(number)
        plt.plot(temp_df["usage"])
    target_path = "my_figures/force_result/{}".format(user_id)
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    if index is not None:
        plt.savefig(target_path + "/main_middle_{}.jpg".format(middles[0][0]))
    else:
        plt.savefig(target_path + "/all_data.jpg")
    plt.close()
