import matplotlib.pyplot as plt
import pandas as pd

from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode

path = "my_data/all_data.csv"
all_df = pd.read_csv(path)

path = "sample_data/hourly_sample_accumulative-usage_gregorian-date.csv"
hourly_df = load_data_frame(path, False, True, FillNanMode.drop)

temp_users = [1502494705221, 1505416405229, 1505125305227] + hourly_df.id.unique().tolist()
print(temp_users)
all_df = all_df[all_df.id.isin(temp_users)]
all_df.to_csv("sample_data/hourly_sample_accumulative-usage_gregorian-date.csv", index=None)
#
# def plot(temp_df: pd.DataFrame):
#     user_id = temp_df['id'].values[0]
#     # resample_values = ["1H", "1D", "7D"]
#     resample_values = ["1D"]
#     for resample_value in resample_values:
#         inner_temp_df = temp_df[['usage']].resample(resample_value).sum()
#         plt.plot(inner_temp_df.index, inner_temp_df.usage, label="usage")
#         plt.tight_layout()
#         plt.legend()
#         plt.savefig("my_figures/usage/{}_{}.jpeg".format(user_id, resample_value))
#         plt.close()
#
#
# hourly_df.set_index("date").groupby("id").apply(plot)
