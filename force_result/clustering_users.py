# from sklearn.metrics import silhouette_score
from tslearn.clustering import TimeSeriesKMeans, silhouette_score

from force_result.three import *


def vector(temp_df: pd.DataFrame):
    temp_df = temp_df.groupby("hour").agg({"usage": "mean"})
    return temp_df.transpose()


df = load_data_frame()
df["hour"] = df.index.hour
df = df.groupby("id").apply(vector)
df = df.reset_index().drop(columns=["level_1"])
users_id = df.id.to_numpy()
df = df.drop(columns=["id"])
x = df.to_numpy()

print(users_id)
print(x)

for i in range(2, 5):
    model = TimeSeriesKMeans(n_clusters=i, metric="softdtw", max_iter=10)
    y_prediction = model.fit_predict(x)

    y_set = set(y_prediction)
    print(y_set)
    for cluster in y_set:
        print("cluster {} count = {}".format(cluster, (y_prediction == cluster).sum()))

    print(f'Silhouette Score(n={i}): {silhouette_score(x, y_prediction, metric="softdtw", sample_size=20, n_jobs=4)}')
