import matplotlib.pyplot as plt
import pandas as pd

# load date set and convert date
data_frame = pd.read_csv("my_data/all_data.csv", date_parser=["datetime"])
data_frame.date = pd.to_datetime(data_frame.date)
data_frame = data_frame[["id", "date", "usage"]]


def plot_user(user_id):
    # select a random user from all users
    user_df = data_frame[data_frame["id"] == user_id]

    # ekhtelaf masraf ha ro hesab mikonim aval to scale haie mokhtalef be darsad
    # user_df["difference_hour"] = 100 * (user_df["usage"] - user_df["usage"].shift(1)) / user_df["usage"]
    print("null count = ", user_df.usage.isnull().sum())
    user_df = user_df.set_index("date")
    user_df["usage"] = user_df["usage"] - user_df["usage"].shift(1)
    half_daily_df = user_df.resample("12H").agg({"usage": "sum"})
    daily_df = user_df.resample("1D").agg({"usage": "sum"})
    week_df = user_df.resample("7D").agg({"usage": "sum"})

    plt.figure()
    plt.subplot(411)
    plt.scatter(user_df.index, user_df["usage"], color='blue', label="daily")

    plt.subplot(412)
    plt.plot(half_daily_df["usage"], color='green')

    plt.subplot(413)
    plt.plot(daily_df["usage"], color='black')

    plt.subplot(414)
    plt.plot(week_df["usage"], color='red', marker='o')
    plt.show()
