import pandas as pd

from force_result.visualization import plot_user


def calculate_usage(temp_df):
    temp_df = temp_df.resample("1H").agg({"usage": "max"})
    temp_df = temp_df.dropna()
    temp_df["usage"] = temp_df["usage"] - temp_df["usage"].shift(1)
    return temp_df


def calculate_bands(temp_df):
    resample_value = "{}H".format(24 * 4)
    moving_avg_windows_size = 50
    std_coe = 6

    # temp_df = temp_df.drop(columns=["id"], axis=0)
    temp_df = temp_df.resample(resample_value).agg({"usage": "sum"})
    temp_df = temp_df.dropna()
    temp_df["avg_value"] = temp_df.usage.rolling(moving_avg_windows_size, min_periods=1).mean()
    temp_df["std_value"] = temp_df.usage.rolling(moving_avg_windows_size, min_periods=1).std()
    temp_df["upper_band"] = temp_df.avg_value + temp_df.std_value * std_coe
    temp_df["lower_band"] = temp_df.avg_value - temp_df.std_value * std_coe
    return temp_df


df = pd.read_csv("my_data/good_data.csv", date_parser=["datetime"])
df.date = pd.to_datetime(df.date)
df = df[["id", "date", "usage"]]
df = df.set_index("date")

df = df.groupby(["id"]).apply(calculate_usage)
df = df.reset_index(level="id", drop=False)

threshold_df = df.groupby("id").resample("1D").agg({"usage": "sum"})
threshold_df = threshold_df.reset_index(level="id", drop=False)
threshold_df = threshold_df.groupby("id").agg({"usage": "mean"})
good_user = threshold_df[threshold_df.usage > 10]
good_user = good_user.reset_index(level="id", drop=False)
good_user_df = df[df.id.isin(good_user.id)]

my_df = df.groupby(["id"]).apply(calculate_bands)
mining_prediction = my_df[my_df.usage > my_df.upper_band]
print(mining_prediction)

mining_prediction = mining_prediction.reset_index(level="id", drop=False)
for index, data in zip(mining_prediction.index, mining_prediction.values):
    plot_user(int(data[0]), index)

# todo bad az detect mean balaie yek chizi bashe
