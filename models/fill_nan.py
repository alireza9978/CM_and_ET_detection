from enum import Enum, auto

import pandas as pd


class FillNanMode(Enum):
    without = auto()
    from_previous_data = auto()
    from_next_data = auto()

    def get_method(self):
        if self == FillNanMode.without:
            return None
        if self == FillNanMode.from_next_data:
            return from_next_data_method
        if self == FillNanMode.from_previous_data:
            return from_previous_data_method


# todo implement methods
def from_next_data_method(temp_df: pd.DataFrame):
    pass


def from_previous_data_method(temp_df: pd.DataFrame):
    pass
