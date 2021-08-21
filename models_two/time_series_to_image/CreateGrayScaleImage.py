import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler

from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode

SEGMENT_LENGTH = 24
path = "my_data/irish.csv"
df = load_data_frame(path, False, False, FillNanMode.linear_auto_fill, True)


def _apply_parallel(data_frame_grouped, func):
    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(func)(group) for name, group in data_frame_grouped)


def make_data_set_single_user(temp_df: pd.DataFrame):
    temp_df = temp_df.reset_index(drop=True)
    user_id = temp_df.id.values[0]

    # apply min max scaler
    scaler_model = MinMaxScaler(feature_range=(0, 255))
    temp_df.usage = scaler_model.fit_transform(temp_df.usage.to_numpy().reshape(-1, 1)).squeeze()

    # select segments with correct length
    temp_df["segment"] = temp_df.index // SEGMENT_LENGTH
    good_segment = temp_df[['usage', 'segment']].groupby("segment").count() == SEGMENT_LENGTH
    temp_df = temp_df[temp_df.segment.isin(good_segment[good_segment.usage].index)]
    temp_df.usage = temp_df.usage.astype(np.uint8)

    # convert each segment to data row
    images = temp_df.usage.to_numpy().reshape(-1, 24, 30)
    name = "GrayScaleImage"
    for i, image in enumerate(images):
        plt.imsave("my_figures/{}/{}_{}.jpeg".format(name, user_id, i), image)
        plt.close()


_apply_parallel(df.groupby("id"), make_data_set_single_user)
