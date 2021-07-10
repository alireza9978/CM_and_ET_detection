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
        plt.savefig("my_figures/mashhad_filter_suspect/{}_{}.jpeg".format(user_id, resample_value))
        plt.close()

