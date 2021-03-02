import pandas as pd


def my_calculate(temp_df):
    resample_value = "{}H".format(24 * 4)
    moving_avg_windows_size = 50
    std_coe = 6

    # temp_df = temp_df.drop(columns=["id"], axis=0)
    temp_df = temp_df.resample(resample_value).agg({"usage": "sum"})
    temp_df = temp_df.dropna()
    temp_df["usage"] = temp_df["usage"] - temp_df["usage"].shift(1)
    temp_df["avg_value"] = temp_df.usage.rolling(moving_avg_windows_size, min_periods=1).mean()
    temp_df["std_value"] = temp_df.usage.rolling(moving_avg_windows_size, min_periods=1).std()
    temp_df["upper_band"] = temp_df.avg_value + temp_df.std_value * std_coe
    temp_df["lower_band"] = temp_df.avg_value - temp_df.std_value * std_coe
    return temp_df


df = pd.read_csv("my_data/good_data.csv", date_parser=["datetime"])
df.date = pd.to_datetime(df.date)
df = df[["id", "date", "usage"]]
df = df.set_index("date")

threshold_df = df.groupby("id").resample("1D").agg({"usage": "sum"})
threshold_df = threshold_df.reset_index(level="id", drop=False)
threshold_df = threshold_df.groupby("id").agg({"usage": "mean"})
good_user = threshold_df[threshold_df.usage > 10]


my_df = df.groupby(["id"]).apply(my_calculate)
# my_df = my_calculate(df_user, resample_value="{}H".format(resample_hour))
# thief_prediction = my_df[my_df.usage < my_df.lower_band]
# my_df_mean = my_df.groupby("id").agg({"usage": "min"})
# print("all user mean = ", my_df_mean.usage.mean())


# my_df = my_df.loc[miner_user.index]
mining_prediction = my_df[my_df.usage > my_df.upper_band]
# middle_index = my_df.index.get_loc(thief_prediction.index[0])
# start = middle_index - 20
# end = middle_index + 20
# if start < 0:
#     start = 0
# if end > len(my_df.index):
#     end = len(my_df.index)
# my_plot(my_df[start:end])
# print(thief_prediction)
print(mining_prediction)
