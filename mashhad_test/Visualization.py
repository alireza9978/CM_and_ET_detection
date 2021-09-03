import jdatetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode


def plot(temp_df: pd.DataFrame, temp_date):
    temp_df = temp_df.reset_index()
    temp_df = temp_df.set_index("date")
    user_id = temp_df['id'].values[0]
    resample_values = ["1H", "1D", "7D"]
    for resample_value in resample_values:
        inner_temp_df = temp_df[['usage']].resample(resample_value).sum()
        plt.plot(inner_temp_df.index, inner_temp_df.usage, label="usage")
        anomal_df = inner_temp_df[inner_temp_df.index > temp_date]
        plt.plot(anomal_df.index, anomal_df.usage, label="usage")
        plt.tight_layout()
        plt.legend()
        plt.savefig("my_figures/mashhad_filter_suspect/{}_{}.jpeg".format(user_id, resample_value))
        plt.close()


def plot_in_one(temp_df: pd.DataFrame, index=""):
    def sample_plot(inner_df: pd.DataFrame):
        user_id = inner_df.id.values[0]
        plt.plot(inner_df.index, inner_df.usage, label=str(user_id))

    temp_df.set_index("date").groupby("id").apply(sample_plot)
    plt.legend()
    plt.savefig("my_figures/test/all_data_{}.jpeg".format(index))
    plt.close()


SEGMENT_LENGTH = 7 * 24
RESAMPLE_VALUE = "1H"
df = load_data_frame("my_data/mashhad_withNan.csv", False, False, FillNanMode.linear_auto_fill)

end_dates = pd.DataFrame([[459502, "1398/05/08"],
                          [7901878, "1398/04/31"],
                          [7700316, "1397/12/22"],
                          [267057, "1398/04/22"],
                          [364909, "1398/04/25"],
                          [500172, "1398/05/05"],
                          [998035, "1398/04/16"],
                          [976926, "1398/04/10"],
                          [647369, "1398/04/22"],
                          [645615, "1398/04/10"],
                          [998240, "1398/04/19"],
                          [7900156, "1398/04/26"],
                          [765802, "1398/04/09"],
                          [643441, "1398/04/25"]], columns=["id", "date"])


def _convert_date_time_hourly(x):
    year = int(str(x)[0:4])
    month = int(str(x)[5:7])
    day = int(str(x)[8:10])
    if month == 12 and day == 30:
        day = 29
    return jdatetime.datetime(year, month, day).togregorian()


end_dates.date = end_dates.date.apply(_convert_date_time_hourly)

for row in end_dates.iterrows():
    plot(df[df.id == row[1][0]], row[1][1])