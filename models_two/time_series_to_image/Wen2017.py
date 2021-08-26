import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode


def transform(temp_df: pd.DataFrame):
    user_id = temp_df.id.values[0]

    usage = temp_df.usage.to_numpy()
    usage_length_sqrt = np.sqrt(usage.shape[0]).astype(np.int)
    selection_rows = np.sort(np.random.choice(usage_length_sqrt, SEGMENT_LENGTH, False))
    l_array = np.zeros(np.square(SEGMENT_LENGTH))
    for l_index, row_index in enumerate(selection_rows):
        start = row_index * SEGMENT_LENGTH
        end = (row_index + 1) * SEGMENT_LENGTH
        l_start = l_index * SEGMENT_LENGTH
        l_end = (l_index + 1) * SEGMENT_LENGTH
        l_array[l_start:l_end] = usage[start:end]

    max_l = np.max(l_array)
    min_l = np.min(l_array)
    out_image = l_array.reshape((SEGMENT_LENGTH, SEGMENT_LENGTH))
    out_image = out_image - min_l
    out_image = out_image / (max_l - min_l)
    out_image = out_image * 255
    out_image = np.round(out_image).astype(np.uint8)

    plt.imsave("my_figures/Wen2017/{}.jpeg".format(user_id), out_image)
    plt.close()


SEGMENT_LENGTH = 64
path = "my_data/irish.csv"
df = load_data_frame(path, False, False, FillNanMode.linear_auto_fill, True)
df.set_index("date").groupby("id").apply(transform)
