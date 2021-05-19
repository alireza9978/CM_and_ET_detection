from enum import Enum, auto

import pandas as pd


class FillNanMode(Enum):
    without = auto()
    drop = auto()
    from_previous_data = auto()
    from_next_data = auto()
    linear_auto_fill = auto()

    def get_method(self):
        if self == FillNanMode.without:
            return None
        if self == FillNanMode.drop:
            return drop_nan_data_method
        if self == FillNanMode.from_next_data:
            return from_next_data_method
        if self == FillNanMode.from_previous_data:
            return from_previous_data_method
        if self == FillNanMode.linear_auto_fill:
            return linear_auto_fill


def _single_user_linear(temp_df: pd.DataFrame):
    temp_df.usage = temp_df.usage.interpolate(method='linear', limit_direction='forward', axis=0).ffill()
    return temp_df


def linear_auto_fill(temp_df: pd.DataFrame):
    return temp_df.groupby("id").apply(_single_user_linear)


def drop_nan_data_method(temp_df: pd.DataFrame):
    return temp_df.dropna()


# todo implement methods
def from_next_data_method(temp_df: pd.DataFrame):
    pass


def from_previous_data_method(temp_df: pd.DataFrame):
    pass
