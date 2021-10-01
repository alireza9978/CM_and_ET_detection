import multiprocessing

import pandas as pd
from joblib import Parallel, delayed


class DetectionParam:
    def __init__(self, resample_value: str = "1D", moving_avg_windows_size: int = 40, std: float = 2.5, m: int = 14,
                 n: int = 10):
        self.resample_value = resample_value
        self.moving_avg_windows_size = moving_avg_windows_size
        self.std_coe = std
        self.m = m
        self.n = n


def _apply_parallel(data_frame_grouped, func):
    result_list = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(func)(group) for name, group in data_frame_grouped)
    return pd.concat(result_list)


class Detection:
    def __init__(self, resample: str, windows_size: int, std: float, new_m: int, new_n: int):
        self.params = DetectionParam(resample, windows_size, std, new_m, new_n)

    def _calculate_bands(self, temp_df: pd.DataFrame):
        temp_df = temp_df.set_index("date")
        my_temp_df = temp_df.resample(self.params.resample_value).agg({"usage": "sum"})
        usage = my_temp_df.usage
        my_temp_df["id"] = temp_df.id[0]
        my_temp_df["mining"] = False
        my_temp_df["theft"] = False
        my_temp_df["to_use"] = True
        start = self.params.moving_avg_windows_size
        while start < usage.shape[0]:
            temp_data = usage[start]
            temp_window = usage.iloc[start - self.params.moving_avg_windows_size:][
                              my_temp_df.iloc[start - self.params.moving_avg_windows_size:]["to_use"]][
                          0:self.params.moving_avg_windows_size]

            temp_avg = temp_window.mean()
            temp_std = temp_window.std()
            temp_upper = temp_avg + (temp_std * self.params.std_coe)
            temp_lower = temp_avg - (temp_std * self.params.std_coe)

            if temp_data > temp_upper:
                step = 1
                while True:
                    m_window = usage[start + step:start + step + self.params.m]
                    m_window_detect = (m_window > temp_upper) & (m_window > 20)
                    m_window_ok = m_window_detect.sum() < self.params.n
                    if m_window_ok:
                        break
                    step += 1
                if step > 1:
                    my_temp_df.iloc[start + 1:start + step + self.params.m, 2] = (
                            usage[start + 1:start + step + self.params.m] > temp_upper)
                    my_temp_df.iloc[start + 1:start + step + self.params.m, 4] = (
                            usage[start + 1:start + step + self.params.m] < temp_upper)
                    start = start + self.params.moving_avg_windows_size + step

            if temp_data < temp_lower:
                step = 1
                while True:
                    m_window = usage[start + step:start + step + self.params.m]
                    m_window_detect = (m_window < temp_lower)
                    m_window_ok = m_window_detect.sum() < self.params.n
                    if m_window_ok:
                        break
                    step += 1
                if step > 1:
                    my_temp_df.iloc[start + 1:start + step + self.params.m, 3] = (
                            usage[start + 1:start + step + self.params.m] < temp_lower)
                    my_temp_df.iloc[start + 1:start + step + self.params.m, 4] = (
                            usage[start + 1:start + step + self.params.m] > temp_lower)
                    start = start + self.params.moving_avg_windows_size + step

            start += 1
        return my_temp_df

    def detect(self, temp_df: pd.DataFrame):
        temp_df = _apply_parallel(temp_df.groupby(["id"]), self._calculate_bands)
        x = temp_df[["mining", "theft", "id"]].reset_index().groupby(["id"]).sum()
        suspect = x[(x.mining > 0) | (x.theft > 0)].reset_index()['id'].unique()
        temp_df = temp_df[temp_df.id.isin(suspect)]
        miners = pd.DataFrame(temp_df[temp_df.mining].id.unique(), columns=["id"])
        miners["mining"] = True
        thefts = pd.DataFrame(temp_df[temp_df.theft].id.unique(), columns=["id"])
        thefts["theft"] = True
        export_df = miners.append(thefts).fillna(False)
        return temp_df, export_df
