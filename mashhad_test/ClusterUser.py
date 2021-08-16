import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.clustering import KernelKMeans

from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode


def vector(temp_df: pd.DataFrame):
    temp_df = temp_df.set_index("date").resample("7D").agg({"usage": "sum"})
    return pd.DataFrame(pd.Series([temp_df.usage.mean(), temp_df.usage.max(), temp_df.usage.min(), temp_df.usage.std(),
                                   temp_df.usage.median()])).transpose()


df = load_data_frame("my_data/mashhad_withNan.csv", False, False, FillNanMode.drop)

feature_df = df.groupby("id").apply(vector)
feature_df = feature_df.fillna(0)
feature_df = feature_df.reset_index()
users_id = feature_df.id
feature_df = feature_df.drop(columns=["id"])

x = feature_df.to_numpy()
clf = StandardScaler()
x = clf.fit_transform(x)


models = [
    KMeans(2),
    KMeans(3),
    KMeans(4),
    KMeans(5),
    KMeans(6),
    KMeans(7),
    KMeans(8),
    KMeans(9),
    DBSCAN(eps=0.5, min_samples=15),
    DBSCAN(eps=0.8, ),
    DBSCAN(eps=0.9, ),
    DBSCAN(min_samples=10),
    DBSCAN(min_samples=3),
    MeanShift(),
]

silhouette_scores = []
for i, model in enumerate(models):
    y_prediction = model.fit_predict(x)

    y_set = set(y_prediction)
    if len(y_set) > 1:
        sil = metrics.silhouette_score(x, y_prediction)
        print(i, sil, len(y_set))
    else:
        print("one cluster")
