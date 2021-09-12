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
        plt.savefig("my_figures/anomalous_users/{}_{}.jpeg".format(user_id, resample_value))
        plt.close()


def plot_in_one(temp_df: pd.DataFrame):
    def sample_plot(inner_df: pd.DataFrame):
        plt.plot(inner_df.index, inner_df.usage)

    temp_df.set_index("date").groupby("id").apply(sample_plot)
    plt.savefig("my_figures/all_data.jpeg")
    plt.close()
