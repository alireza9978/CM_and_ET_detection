from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.utils import to_time_series_dataset
from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode
from models.filters import data_frame_agg

df = load_data_frame("my_data/mashhad_withNan.csv", False, False, FillNanMode.drop)
df = data_frame_agg(df, "7D")
x = to_time_series_dataset(df.groupby("id").apply(lambda temp_df: temp_df.usage.to_numpy().reshape(-1, 1)).to_numpy())
models = [
    TimeSeriesKMeans(n_clusters=2, metric="softdtw"),
    TimeSeriesKMeans(n_clusters=3, metric="softdtw"),
    TimeSeriesKMeans(n_clusters=4, metric="softdtw"),
    TimeSeriesKMeans(n_clusters=5, metric="softdtw"),
]

for i, model in enumerate(models):
    y_prediction = model.fit_predict(x)

    y_set = set(y_prediction)
    if len(y_set) > 1:
        sil = silhouette_score(x, y_prediction, metric="dtw")
        print(i, sil, len(y_set))
    else:
        print("one cluster")
