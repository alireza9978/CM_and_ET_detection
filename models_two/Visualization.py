import matplotlib.pyplot as plt
import pandas as pd


def plot(temp_df: pd.DataFrame):
    user_id = temp_df['id'].values[0]
    resample_values = ["1H", "1D", "7D"]
    for resample_value in resample_values:
        inner_temp_df = temp_df[['usage']].resample(resample_value).sum()
        plt.plot(inner_temp_df.index, inner_temp_df.usage, label="usage")
        plt.tight_layout()
        plt.legend()
        plt.savefig("my_figures/irish/{}_{}.jpeg".format(user_id, resample_value))
        plt.close()


def plot_anomaly(temp_df: pd.DataFrame, real="real"):
    user_id = temp_df['id'].values[0]
    resample_values = ["1H", "1D", "7D"]
    for resample_value in resample_values:
        inner_temp_df = temp_df[['usage']].resample(resample_value).sum()
        plt.plot(inner_temp_df.index, inner_temp_df.usage, label="usage")
        plt.tight_layout()
        plt.legend()
        plt.savefig("my_figures/irish/{}_{}_{}.jpeg".format(user_id, resample_value, real))
        plt.close()


def plot_mean_usage(temp_df: pd.DataFrame):
    user_id = temp_df['id'].values[0]
    temp_df["hour"] = temp_df.index.hour
    temp_df = temp_df[["usage", "hour"]].groupby("hour").mean()
    plt.plot(temp_df.index, temp_df.usage, label="usage_mean")
    plt.tight_layout()
    plt.legend()
    plt.savefig("my_figures/irish/24_hour_mean_{}.jpeg".format(user_id))
    plt.close()


if __name__ == '__main__':
    path = "/mnt/79e06c5d-876b-45fd-a066-c9aac1a1c932/Dataset/Power Distribution/irish.csv"
    df = next(pd.read_csv(path, chunksize=100000, converters={"date": pd.to_datetime}))
    # df = pd.read_csv(path, converters={"date": pd.to_datetime})

    df.set_index("date").groupby("id").apply(plot)
    df.set_index("date").groupby("id").apply(plot_mean_usage)
