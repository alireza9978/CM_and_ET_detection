import numpy as np
import pandas as pd
from tslearn.metrics import dtw
from tslearn.utils import to_time_series_dataset

from mashhad_test.Visualization import plot_in_one
from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode
from models.filters import data_frame_agg

SEGMENT_LENGTH = 7 * 24
RESAMPLE_VALUE = "1H"
df = load_data_frame("/mnt/79e06c5d-876b-45fd-a066-c9aac1a1c932/Dataset/Power Distribution/irish.csv", False, False,
                     FillNanMode.without, small_part=True)
temp_users_id = np.random.choice(df.id.unique(), 100)
df = df[df.id.isin(temp_users_id)]
df = data_frame_agg(df, "7D")

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

result = pd.DataFrame(result, columns=["one", "two", "distance"])
similar_indexes = result.sort_values("distance").values[:10, :2]
# df = df[df.id.isin(result[result[:, 2].argmin()][:-1])]
index = 0
for row in similar_indexes:
    similar_user_df = df[df.id.isin(row)]
    plot_in_one(similar_user_df, str(index))
    index += 1
