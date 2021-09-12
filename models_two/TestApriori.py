import numpy as np
import pandas as pd

from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode
from models.filters import select_random_user, data_frame_agg

path = "/mnt/79e06c5d-876b-45fd-a066-c9aac1a1c932/Dataset/Power Distribution/irish.csv"
df = load_data_frame(path, False, False, FillNanMode.linear_auto_fill, True)
df, user_id = select_random_user(df)
df = data_frame_agg(df, "1D")
df = df.drop(columns=["id"])
df = df.reset_index(drop=True)
df_usage = np.diff(df.usage)
df_usage = np.round(df_usage, 1)

time_steps = 7


# Generated training sequences for use in the model.
def create_sequences(values):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i: (i + time_steps)])
    return np.stack(output)


cols = [str(i) for i in range(time_steps)]
transactions = create_sequences(df_usage)
print(transactions.shape)
df = pd.DataFrame(transactions, columns=cols)
df["temp"] = 1
counts = df.groupby(cols).count().sort_values("temp", ascending=False)
print(counts[:10])

