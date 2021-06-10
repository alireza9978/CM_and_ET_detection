import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler

SEGMENT_LENGTH = 30
RESAMPLE_VALUE = "1D"
df = pd.read_csv("my_data/small_irish_anomaly.csv")


def _apply_parallel(data_frame_grouped, func):
    result_list = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(func)(group) for name, group in data_frame_grouped)
    return pd.concat(result_list)


def make_data_set_single_segment(temp_df: pd.DataFrame):
    temp_df = temp_df.reset_index(drop=True)
    anomaly_count = temp_df.anomaly.sum()
    temp_df = temp_df[["usage"]].T
    if 0 < anomaly_count < 30:
        temp_df['label'] = 1
    else:
        temp_df['label'] = 0
    return temp_df


def make_data_set_single_user(temp_df: pd.DataFrame):
    temp_df = temp_df.reset_index(drop=True)

    # resample usage
    temp_df.date = pd.to_datetime(temp_df.date)
    temp_df = temp_df.set_index("date").resample(RESAMPLE_VALUE).agg({"usage": "sum", "anomaly": np.any})
    temp_df = temp_df.reset_index(drop=True)

    # using fourier
    transformed = np.fft.fft(temp_df.usage.to_numpy())
    transformed[temp_df.shape[0] // 20:] = 0
    temp_df.usage = np.abs(np.fft.ifft(transformed))

    # apply min max scaler
    scaler_model = MinMaxScaler()
    temp_df.usage = scaler_model.fit_transform(temp_df.usage.to_numpy().reshape(-1, 1)).squeeze()

    # select segments with correct length
    temp_df["segment"] = temp_df.index // SEGMENT_LENGTH
    good_segment = temp_df[['usage', 'segment']].groupby("segment").count() == SEGMENT_LENGTH
    temp_df = temp_df[temp_df.segment.isin(good_segment[good_segment.usage].index)]

    # convert each segment to data row
    segments = temp_df.groupby("segment").apply(make_data_set_single_segment)

    return segments.reset_index(drop=True)


data_set = df.groupby("id").apply(make_data_set_single_user)
# data_set = _apply_parallel(df.groupby("id"), make_data_set_single_user)
print(data_set)
