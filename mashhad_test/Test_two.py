import pandas as pd
from tslearn.metrics import dtw_path
from tslearn.utils import to_time_series_dataset

from mashhad_test.Visualization import plot_in_one
from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode
from models.filters import data_frame_agg

SEGMENT_LENGTH = 7 * 24
RESAMPLE_VALUE = "1H"
df = load_data_frame("my_data/mashhad_withNan.csv", False, False, FillNanMode.linear_auto_fill)
df = data_frame_agg(df, "7D")
# temp_users_id = np.random.choice(df.id, 10)
# df = df[df.id.isin(temp_users_id)]

# def calculate_pairwise_distance(temp_df: pd.DataFrame):
#     user_id = temp_df.id.values[0]
#     my_small_df = df[~df.id.isin(user_id)]
#
#
#     print(temp_df)

ts_df = to_time_series_dataset(df[["usage", "id"]].groupby("id").apply(lambda x: x.to_numpy().reshape(-1)).to_numpy())
users_id = df.id.unique()
index = 0
result = []
for one in ts_df:
    inner_index = index + 1
    for two in ts_df[index + 1:]:
        result.append([users_id[index], users_id[inner_index], dtw(one, two)])
        inner_index += 1
    index += 1
