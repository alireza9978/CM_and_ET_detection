import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode
from models_two.AddAnomaly import reduce_usage_randomly

path = "/mnt/79e06c5d-876b-45fd-a066-c9aac1a1c932/Dataset/Power Distribution/irish.csv"


def plot_differences():
    shift_value = 7
    df = load_data_frame(path, False, False, FillNanMode.linear_auto_fill, True)
    anomaly_df = df.set_index("date").groupby("id").apply(reduce_usage_randomly).reset_index()
    df = df.set_index("date").groupby("id").apply(lambda x: x.usage - x.usage.shift(shift_value)).dropna().reset_index()
    anomaly_df = anomaly_df.set_index("date").groupby("id").apply(
        lambda x: x.usage - x.usage.shift(shift_value)).dropna().reset_index()

    temp_id = np.random.choice(df["id"].unique())

    df[df.id == temp_id]["usage"].plot()
    anomaly_df[anomaly_df.id == temp_id]["usage"].plot()
    plt.show()
    plt.close()


def reduce_size():
    shift_value = 7
    df = load_data_frame(path, False, False, FillNanMode.linear_auto_fill, True)
    anomaly_df = df.set_index("date").groupby("id").apply(reduce_usage_randomly).reset_index()

    def inner_reduction(inner_df: pd.DataFrame):
        inner_df.usage = inner_df.usage - inner_df.usage.shift(1)
        inner_df = inner_df.dropna().reset_index()
        inner_df["segment"] = inner_df.index % shift_value
        result = inner_df[["usage", "segment"]].groupby("segment").mean()
        result.columns = ["mean"]
        result["var"] = inner_df[["usage", "segment"]].groupby("segment").var()
        return result

    df = df.set_index("date").groupby("id").apply(inner_reduction).reset_index()
    anomaly_df = anomaly_df.set_index("date").groupby("id").apply(inner_reduction).dropna().reset_index()
    temp_id = np.random.choice(df["id"].unique())

    df[df.id == temp_id]["mean"].plot()
    anomaly_df[anomaly_df.id == temp_id]["mean"].plot()
    plt.show()
    plt.close()
    df[df.id == temp_id]["var"].plot()
    anomaly_df[anomaly_df.id == temp_id]["var"].plot()
    plt.show()
    plt.close()


if __name__ == '__main__':
    reduce_size()
    plot_differences()
