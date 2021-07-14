import random

import pandas as pd

from models_two.Visualization import plot_anomaly


def reduce_usage_randomly(temp_df: pd.DataFrame):
    start = temp_df.index[0]
    end = temp_df.index[-1]
    date_range = end - start
    anomaly_start = start + (date_range / 4)

    # 0.2 ~ 0.5
    reduction_coe = (random.random() * 0.3) + 0.2
    temp_start = anomaly_start + ((end - anomaly_start) * random.random() * 0.5)
    temp_end = temp_start + ((end - anomaly_start) * random.random())
    if (temp_end - temp_start) < pd.to_timedelta(60, "d"):
        temp_end += pd.to_timedelta(60, "d")
    temp_end = min(temp_end, end)

    temp_df.loc[temp_start:temp_end, 'usage'] = temp_df.loc[temp_start:temp_end, 'usage'] * reduction_coe
    temp_df["anomaly"] = False
    temp_df.loc[temp_start:temp_end, "anomaly"] = True

    return temp_df


if __name__ == '__main__':
    path = "/mnt/79e06c5d-876b-45fd-a066-c9aac1a1c932/Dataset/Power Distribution/irish.csv"
    df = pd.read_csv(path, converters={"date": pd.to_datetime})
    # df = pd.read_csv(path, chunksize=100000, converters={"date": pd.to_datetime})

    anomaly_df = df.set_index("date").groupby("id").apply(reduce_usage_randomly)
    for user in list(anomaly_df.id.unique()):
        user_df = anomaly_df[anomaly_df.id == user]
        plot_anomaly(user_df, "anomaly")
        user_df = df[df.id == user].set_index("date")
        plot_anomaly(user_df, "clean")

    anomaly_df.to_csv("my_data/irish_anomaly.csv", index=None)
