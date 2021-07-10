import os

import pandas as pd


def load_file():
    path = "smart_data/"
    result_df = pd.DataFrame()
    for file_name in os.listdir(path):
        user_id = file_name.split(".")[0]
        file_path = path + file_name
        temp_df = pd.read_excel(file_path, header=None)
        temp_df["id"] = user_id
        temp_df.columns = ["date", "time", "usage", "id"]
        temp_df["date"] = temp_df.date.astype(str) + " " + temp_df.time.astype(str)
        temp_df['date'] = pd.to_datetime(temp_df.date)
        result_df = result_df.append(temp_df[["id", "date", "usage"]])

    return result_df.reset_index(drop=True)


df = load_file()
df = df.set_index("date").groupby("id").resample("1H")[["usage"]].sum().reset_index()
df.to_csv("my_data/mashhad.csv", index=None)
