import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import MinMaxScaler, Normalizer

from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode

SEGMENT_LENGTH = 7 * 24
RESAMPLE_VALUE = "1H"
df = load_data_frame("my_data/mashhad_withNan.csv", False, False, FillNanMode.linear_auto_fill)


def _apply_parallel(data_frame_grouped, func):
    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(func)(group) for name, group in data_frame_grouped)


def make_data_set_single_user(temp_df: pd.DataFrame):
    temp_df = temp_df.reset_index(drop=True)
    user_id = temp_df.id.values[0]
    # resample usage
    temp_df.date = pd.to_datetime(temp_df.date)
    temp_df = temp_df.set_index("date").resample(RESAMPLE_VALUE).agg({"usage": "sum"})
    temp_df = temp_df.reset_index(drop=True)

    # map to gaussian
    scaler_model = Normalizer(norm='l2')
    temp_df.usage = scaler_model.fit_transform(temp_df.usage.to_numpy().reshape(1, -1)).squeeze()

    # apply min max scaler
    scaler_model = MinMaxScaler(feature_range=(0, 255))
    temp_df.usage = scaler_model.fit_transform(temp_df.usage.to_numpy().reshape(-1, 1)).squeeze()


    # select segments with correct length
    temp_df["segment"] = temp_df.index // SEGMENT_LENGTH
    good_segment = temp_df[['usage', 'segment']].groupby("segment").count() == SEGMENT_LENGTH
    temp_df = temp_df[temp_df.segment.isin(good_segment[good_segment.usage].index)]
    temp_df.usage = temp_df.usage.astype(np.uint8)

    # convert each segment to data row
    image = temp_df.usage.to_numpy().reshape(-1, SEGMENT_LENGTH)
    labels = np.arange(0, SEGMENT_LENGTH + 1, 24)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(image)
    ax.set_xticks(labels)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.xticks(labels, [str(x) for x in labels])
    plt.colorbar(im, cax=cax)
    plt.savefig("my_figures/mashhad_gray/{}.jpeg".format(user_id))
    plt.close()


# df.groupby("id").apply(make_data_set_single_user)
_apply_parallel(df.groupby("id"), make_data_set_single_user)
