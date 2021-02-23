import pandas as pd

data_frame = pd.read_csv("my_data/all_data.csv", date_parser=["datetime"])
data_frame.date = pd.to_datetime(data_frame.date)
data_frame = data_frame[["id", "date", "usage"]]
data_frame = data_frame.set_index("date")


def calculate_difference(temp_df):
    temp_df["difference_hour"] = temp_df["usage"] - temp_df["usage"].shift(1)
    return temp_df


# calculate difference of accumulative usage
data_frame = data_frame.groupby(["id"]).apply(calculate_difference)
# show each user usage in day
data_frame = data_frame.groupby(["id"]).resample("D").aggregate({"difference_hour": lambda x: x.tolist()})
data_frame = data_frame.reset_index(drop=False)
# rename columns
data_frame = data_frame.rename(columns={'id': 'variable', 'difference_hour': 'power'})
# convert timestamps to date
data_frame["date"] = data_frame["date"].apply(lambda x: x.date())
# change columns order
data_frame = data_frame[data_frame.columns[[1, 0, 2]]]

print("saving")
data_frame.to_csv("my_data/spark_readable.csv")
