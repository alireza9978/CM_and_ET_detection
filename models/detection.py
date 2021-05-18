import multiprocessing

import pandas as pd
from joblib import Parallel, delayed


def _calculate_bands(temp_df: pd.DataFrame):
    temp_df = temp_df.set_index("date")
    my_temp_df = temp_df.resample(resample_value).agg({"usage": "sum"})
    usage = my_temp_df.usage
    my_temp_df["id"] = temp_df.id[0]
    my_temp_df["mining"] = False
    my_temp_df["theft"] = False
    my_temp_df["to_use"] = True
    start = moving_avg_windows_size
    while start < usage.shape[0]:
        temp_data = usage[start]
        temp_window = usage.iloc[start - moving_avg_windows_size:][
                          my_temp_df.iloc[start - moving_avg_windows_size:]["to_use"]][0:moving_avg_windows_size]

        temp_avg = temp_window.mean()
        temp_std = temp_window.std()
        temp_upper = temp_avg + (temp_std * std_coe)
        temp_lower = temp_avg - (temp_std * std_coe)

        if temp_data > temp_upper:
            step = 1
            while True:
                m_window = usage[start + step:start + step + m]
                m_window_detect = (m_window > temp_upper) & (m_window > 20)
                m_window_ok = m_window_detect.sum() < n
                if m_window_ok:
                    break
                step += 1
            if step > 1:
                my_temp_df.iloc[start + 1:start + step + m, 2] = (usage[start + 1:start + step + m] > temp_upper)
                my_temp_df.iloc[start + 1:start + step + m, 4] = (usage[start + 1:start + step + m] < temp_upper)
                start = start + moving_avg_windows_size + step

        if temp_data < temp_lower:
            step = 1
            while True:
                m_window = usage[start + step:start + step + m]
                m_window_detect = (m_window < temp_lower)
                m_window_ok = m_window_detect.sum() < n
                if m_window_ok:
                    break
                step += 1
            if step > 1:
                my_temp_df.iloc[start + 1:start + step + m, 3] = (usage[start + 1:start + step + m] < temp_lower)
                my_temp_df.iloc[start + 1:start + step + m, 4] = (usage[start + 1:start + step + m] > temp_lower)
                start = start + moving_avg_windows_size + step

        start += 1
    return my_temp_df


def _apply_parallel(data_frame_grouped, func):
    result_list = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(func)(group) for name, group in data_frame_grouped)
    return pd.concat(result_list)


def detect(temp_df: pd.DataFrame):
    temp_df = _apply_parallel(temp_df.groupby(["id"]), _calculate_bands)
    return temp_df


def set_detection_parameters(resample: str, windows_size: int, std: float, new_m: int, new_n: int):
    global resample_value
    global moving_avg_windows_size
    global std_coe
    global m
    global n
    resample_value = resample
    moving_avg_windows_size = windows_size
    std_coe = std
    m = new_m
    n = new_n


resample_value = "{}H".format(24)
moving_avg_windows_size = 40
std_coe = 2.5
m = 20
n = 15
