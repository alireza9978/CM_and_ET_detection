import os

import matplotlib.pyplot as plt
import pandas as pd

# load date set and convert date
data_frame = pd.read_csv("my_data/all_data.csv", date_parser=["datetime"])
data_frame.date = pd.to_datetime(data_frame.date)
data_frame = data_frame[["id", "date", "usage"]]


def plot_user(user_id, index: pd.Timestamp = None):
    # select a random user from all users
    user_df = data_frame[data_frame["id"] == user_id]

    # nazdik tarin index be index vorodi ro to list peida mikone
    def nearest(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))

    # ekhtelaf masraf ha ro hesab mikonim aval to scale haie mokhtalef be darsad
    # user_df["difference_hour"] = 100 * (user_df["usage"] - user_df["usage"].shift(1)) / user_df["usage"]
    user_df = user_df.set_index("date")
    user_df["usage"] = user_df["usage"] - user_df["usage"].shift(1)
    half_daily_df = user_df.resample("12H").agg({"usage": "sum"})
    daily_df = user_df.resample("1D").agg({"usage": "sum"})
    week_df = user_df.resample("7D").agg({"usage": "sum"})

    if index is not None:
        main_middle = user_df.index.get_loc(nearest(user_df.index, index))
        half_day_middle = half_daily_df.index.get_loc(nearest(half_daily_df.index, index))
        day_middle = daily_df.index.get_loc(nearest(daily_df.index, index))
        week_middle = week_df.index.get_loc(nearest(week_df.index, index))

        middles = [(main_middle, len(user_df.index), 240), (half_day_middle, len(half_daily_df.index), 20),
                   (day_middle, len(daily_df.index), 10), (week_middle, len(week_df.index), 10)]
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
        user_df = user_df[start[0]:end[0]]
        half_daily_df = half_daily_df[start[1]:end[1]]
        daily_df = half_daily_df[start[3]:end[2]]
        week_df = half_daily_df[start[2]:end[3]]

    plt.figure()
    plt.subplot(411)
    plt.plot(user_df["usage"], color='blue', label="hourly")

    plt.subplot(412)
    plt.plot(half_daily_df["usage"], color='green', label="half_day")

    plt.subplot(413)
    plt.plot(daily_df["usage"], color='black', label="daily")

    plt.subplot(414)
    plt.plot(week_df["usage"], color='red', marker='o', label="weekly")
    target_path = "my_figures/force_result/{}".format(user_id)
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    if index is not None:
        plt.savefig(target_path + "/main_middle_{}.jpg".format(middles[0][0]))
    else:
        plt.savefig(target_path + "/all_data.jpg")
