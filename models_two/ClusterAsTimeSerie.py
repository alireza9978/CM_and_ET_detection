from tslearn.clustering.utils import to_time_series_dataset
from tslearn.metrics import dtw, soft_dtw
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode

path = "my_data/irish.csv"
df = load_data_frame(path, False, False, FillNanMode.linear_auto_fill, small_part=True)
df = to_time_series_dataset(df.groupby("id").apply(lambda x: x.usage.to_list()).to_list())


