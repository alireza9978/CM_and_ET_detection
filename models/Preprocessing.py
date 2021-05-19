import multiprocessing

import jdatetime
import pandas as pd
from joblib import Parallel, delayed

from models.exception import HourlyDateException, MonthlyDateException, WrongColumnsException, WrongDateFormatException
from models.fill_nan import FillNanMode


def load_data_frame(path: str, persian_date: bool, accumulative_usage: bool, fill_nan: FillNanMode):
    temp_df = pd.read_csv(path)
    if len(temp_df.columns) == 3:
        return _load_data_frame_hourly(temp_df, persian_date, accumulative_usage, fill_nan)
    if len(temp_df.columns) == 6:
        return _load_data_frame_monthly(temp_df, persian_date, fill_nan)
    raise WrongColumnsException()


def _load_data_frame_hourly(temp_df: pd.DataFrame, persian_date: bool, accumulative_usage: bool, fill_nan: FillNanMode):
    # reordering columns
    temp_df = temp_df[["id", "date", "usage"]]

    # todo try-catch for date error
    # convert date to gregorian
    if persian_date:
        try:
            temp_df['date'] = temp_df['date'].apply(lambda x: _convert_date_time_hourly(x))
        except:
            raise HourlyDateException()
    # convert date to pandas datetime
    temp_df.date = pd.to_datetime(temp_df.date)

    # sort values of data set
    temp_df = temp_df.sort_values(by=["id", "date"])

    # fill nan based on implemented method
    fill_nan_method = fill_nan.get_method()
    if fill_nan_method is not None:
        temp_df = fill_nan_method(temp_df)

    # convert usage to Non-cumulative
    if accumulative_usage:
        temp_df = temp_df.groupby(["id"]).apply(_calculate_usage_hourly)

    # reset index and return clean data set
    temp_df = temp_df.reset_index(drop=False)
    # reordering columns
    temp_df = temp_df[["id", "date", "usage"]]
    return temp_df


def _calculate_usage_hourly(temp_df: pd.DataFrame):
    # set date as index for usage operation
    temp_df = temp_df.set_index("date")

    # make all data set frequency 1 hour
    temp_df["usage"] = temp_df.resample("1H").agg({"usage": "max"})

    temp_df["usage"] = temp_df["usage"] - temp_df["usage"].shift(1)
    temp_df = temp_df.iloc[1:]["usage"]

    return temp_df


def _load_data_frame_monthly(temp_df: pd.DataFrame, persian_date: bool, fill_nan: FillNanMode):
    # sort values of data set
    temp_df = temp_df.sort_values(by=["id", "start_date"])

    # todo try-catch for date error
    # convert date to gregorian
    if persian_date:
        try:
            temp_df['start_date'] = temp_df['start_date'].apply(lambda x: _convert_date_time_monthly(x))
            temp_df['end_date'] = temp_df['end_date'].apply(lambda x: _convert_date_time_monthly(x))
        except:
            raise MonthlyDateException()

    # convert date to pandas datetime
    temp_df["start_date"] = pd.to_datetime(temp_df["start_date"])
    temp_df["end_date"] = pd.to_datetime(temp_df["end_date"])

    # generate column "days" that shows length of each cycle
    temp_df["days"] = temp_df["end_date"] - temp_df["start_date"]
    temp_df["days"] = temp_df["days"].dt.days

    # remove user that have cycle with length less than 1 day
    temp_df = temp_df[~temp_df.id.isin(temp_df[temp_df.days < 1].id.unique())]

    # fill nan based on implemented method
    fill_nan_method = fill_nan.get_method()
    if fill_nan_method is not None:
        temp_df = fill_nan_method(temp_df)

    temp_df = _clean_data(temp_df)

    # reset index and return clean data set
    temp_df = temp_df.reset_index(drop=False)
    # reordering columns
    temp_df = temp_df[["id", "date", "usage"]]
    return temp_df


def _calculate_usage_monthly(temp_df: pd.DataFrame):
    temp_df["low"] = temp_df["low"] - temp_df["low"].shift(1)
    temp_df["medium"] = temp_df["medium"] - temp_df["medium"].shift(1)
    temp_df["high"] = temp_df["high"] - temp_df["high"].shift(1)
    return temp_df


def _clean_data(temp_df: pd.DataFrame):
    return _apply_parallel(temp_df.groupby(['id']), _clean_user_df)


def _apply_parallel(data_frame_grouped, func):
    result_list = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(func)(group) for name, group in data_frame_grouped)
    return pd.concat(result_list)


def _clean_user_df(inner_df: pd.DataFrame):
    indexes = pd.date_range(inner_df.start_date.min(), inner_df.end_date.max(), freq='D', name='date')
    out_df = pd.DataFrame(index=indexes, columns=["usage"])
    for temp_i, temp_row in inner_df.iterrows():
        start = out_df.index.get_loc(temp_row.start_date)
        end = out_df.index.get_loc(temp_row.end_date)
        value = (temp_row["low"] + temp_row["medium"] + temp_row["high"]) / temp_row["days"]
        out_df.iloc[start:end, 0] = out_df.iloc[start:end, 0].fillna(0) + value

    out_df = out_df.resample("M").sum()
    out_df["id"] = inner_df.id.values[0]
    return out_df


def _convert_date_time_hourly(x):
    # expected format %Y-%M-%d %h:%m:%s
    try:
        year = int(str(x)[0:4])
        month = int(str(x)[5:7])
        day = int(str(x)[8:10])
        hour = int(str(x)[11:13])
        minute = int(str(x)[14:16])
        second = int(str(x)[17:19])
        if month == 12 and day == 30:
            day = 29
        return jdatetime.datetime(year, month, day, hour, minute, second).togregorian()
    except:
        raise WrongDateFormatException()


def _convert_date_time_monthly(x):
    # expected format %Y-%M-%d
    try:
        year = int(str(x)[0:4])
        month = int(str(x)[5:7])
        day = int(str(x)[8:10])
        if month == 12 and day == 30:
            day = 29
        return jdatetime.date(year, month, day).togregorian()
    except:
        raise WrongDateFormatException()
