import datetime
from pathlib import Path
import pandas as pd


def load_file():
    path = "H:/Projects/Pycharm/Datasets/Power Distribution/Irish/Files/CER Electricity Revised March 2012/"
    result_df = pd.DataFrame()
    for i in range(1, 7):
        test_path = Path(path + "File{}.txt".format(i))
        # temp_df = next(pd.read_csv(test_path, sep=" ", header=None, chunksize=50000))
        temp_df = pd.read_csv(test_path, sep=" ", header=None)
        temp_df.columns = ["id", "date", "usage"]
        temp_df["time"] = temp_df.date.mod(100)
        temp_df["date"] = temp_df.date // 100
        temp_df["date"] = pd.to_datetime(datetime.date(2009, 1, 1)) + \
                          pd.to_timedelta(temp_df.date, unit="day") + \
                          pd.to_timedelta((temp_df["time"] - 1) * 30, unit="minute")
        result_df = result_df.append(temp_df[["id", "date", "usage"]])

    return result_df.reset_index(drop=True)


df = load_file()
df = df.set_index("date").groupby("id").resample("1H")[["usage"]].sum().reset_index()
df.to_csv("my_data/irish.csv", index=None)
