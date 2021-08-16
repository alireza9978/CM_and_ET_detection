import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable

from models_two.Visualization import plot_anomaly
from tslearn.metrics import dtw
from tslearn.utils import to_time_series_dataset
import matplotlib.pyplot as plt
from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode
from models.filters import data_frame_agg

SEGMENT_LENGTH = 24 * 7
SEGMENT_SIZE = 24
RESAMPLE_VALUE = "1H"
df = load_data_frame("/mnt/79e06c5d-876b-45fd-a066-c9aac1a1c932/Dataset/Power Distribution/irish.csv", False, False,
                     FillNanMode.without, small_part=True)

temp_users_id = np.random.choice(df.id.unique(), 1)
df = df[df.id.isin(temp_users_id)]
df = data_frame_agg(df, RESAMPLE_VALUE)

df = df.reset_index(drop=True)

df["segment"] = df.index // SEGMENT_LENGTH
good_segment = df[['usage', 'segment']].groupby("segment").count() == SEGMENT_LENGTH
df = df[df.segment.isin(good_segment[good_segment.usage].index)]

ts_df = to_time_series_dataset(df.usage.to_numpy().reshape(-1, SEGMENT_SIZE))

step = 7
index = 0
result = []
for i in range(ts_df.shape[0] - step):
    result.append(dtw(ts_df[i], ts_df[i + step]))


result = np.array(result).reshape(-1, 7)

scaler_model = MinMaxScaler(feature_range=(0, 255))
result = scaler_model.fit_transform(result)

fig, ax = plt.subplots(1, 1, figsize=(5, 10))
im = ax.imshow(result)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.savefig("my_figures/test_three/dtw_distance.jpeg")
plt.close()

df.set_index("date").groupby("id").apply(plot_anomaly)