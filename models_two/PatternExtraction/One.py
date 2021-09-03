import numpy as np
import numpy as np
import pandas as pd

from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode
from models.filters import data_frame_agg
from models_two.AddAttacksToUser import attack1, attack2, attack3, attack4, attack5, attack6
from models_two.ClusterUser import clustering

SEGMENT_LENGTH = 7
path = "/mnt/79e06c5d-876b-45fd-a066-c9aac1a1c932/Dataset/Power Distribution/irish.csv"
df = load_data_frame(path, False, False, FillNanMode.linear_auto_fill, True)

attacks = [attack1, attack2, attack3, attack4, attack5, attack6]
user_data_points_count = df[["id", "usage"]].groupby("id").count()
proper_users_ids = (user_data_points_count[user_data_points_count >= user_data_points_count.mean()].dropna()).index
df = df[df.id.isin(proper_users_ids)]


def vectorization(temp_df: pd.DataFrame):
    temp_df = temp_df.reset_index()

    temp_df["segment"] = temp_df.index // SEGMENT_LENGTH
    good_segment = temp_df[['usage', 'segment']].groupby("segment").count() == SEGMENT_LENGTH
    temp_df = temp_df[temp_df.segment.isin(good_segment[good_segment.usage].index)]

    temp_df = temp_df.usage.to_numpy().reshape(-1, SEGMENT_LENGTH)
    temp_df = np.round(temp_df, 1)

    return temp_df


df = data_frame_agg(df, "1D")
temp_vec = df.groupby("id").apply(vectorization)
temp_vec = temp_vec.reset_index()
temp_vec.columns = ["id", "values"]

temp_ids = np.array([])
for user_id, row in temp_vec.iterrows():
    temp_ids = np.concatenate([temp_ids, np.full(row[1].shape[0], row[0])])

temp_vec = np.concatenate(temp_vec.values)

sampled_index = np.random.choice(np.arange(temp_vec.shape[0]), int(0.3 * temp_vec.shape[0]), replace=False)
sampled_data = temp_vec[sampled_index]
sampled_id = temp_ids[sampled_index]
clustering(sampled_data, range(sampled_data.shape[0]))
