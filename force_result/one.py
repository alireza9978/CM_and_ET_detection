import matplotlib.pyplot as plt
import pandas as pd

from force_result.visualization import plot_user


def my_plot(data_frame):
    fig, ax = plt.subplots()
    ax.plot(data_frame.usage, 'b', marker='*', linestyle='-', linewidth=0.5, label='Usage')
    ax.plot(data_frame.avg_value, 'c', marker='.', linestyle='-', label='AVG')
    ax.plot(data_frame.upper_band, 'r', marker='.', linestyle='-', label='UP')
    ax.plot(data_frame.lower_band, 'g', marker='.', linestyle='-', label='DOWN')
    ax.set_ylabel('KWH')
    ax.legend()
    plt.savefig("my_figures/temp.jpg")


def my_calculate(temp_df, resample_value="12H", moving_avg_windows_size=10, std_coe=2.5):
    temp_df = temp_df.drop(columns=["id"], axis=0)
    temp_df = temp_df.resample(resample_value).agg({"usage": "sum"})
    temp_df = temp_df.dropna()
    # temp_df["usage"] = temp_df["usage"] - temp_df["usage"].shift(1)
    temp_df["avg_value"] = temp_df.usage.rolling(moving_avg_windows_size, min_periods=1).mean()
    temp_df["std_value"] = temp_df.usage.rolling(moving_avg_windows_size, min_periods=1).std()
    temp_df["upper_band"] = temp_df.avg_value + temp_df.std_value * std_coe
    temp_df["lower_band"] = temp_df.avg_value - temp_df.std_value * std_coe
    return temp_df


df = pd.read_csv("my_data/good_data.csv", date_parser=["datetime"])
df.date = pd.to_datetime(df.date)
df = df[["id", "date", "usage"]]
df = df.set_index("date")

# temp_user = df["id"][random.randint(0, df["id"].shape[0])]
temp_user = 1502292505220
df_user = df[df["id"] == temp_user]
df_user = df_user.groupby("id").resample("1H").agg({"usage": "max"})
df_user = df_user.reset_index(level="id", drop=False)
df_user["usage"] = df_user["usage"] - df_user["usage"].shift(1)

resample_hour = 24
my_df = my_calculate(df_user, resample_value="{}H".format(resample_hour), moving_avg_windows_size=24, std_coe=2.5)

thief_prediction = my_df[my_df.usage < my_df.lower_band]
mining_prediction = my_df[my_df.usage > my_df.upper_band]

if mining_prediction.shape[0] > 0 or thief_prediction.shape[0] > 0:
    print(thief_prediction)
    print(mining_prediction)
    for index in mining_prediction.index:
        plot_user(temp_user, index)
    for index in thief_prediction.index:
        plot_user(temp_user, index)
    plot_user(temp_user)
else:
    print("not thief found")
