import pickle

from k_means_constrained import KMeansConstrained
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from force_result.three import *


def vector(temp_df: pd.DataFrame):
    temp_df = temp_df.resample("7D").agg({"usage": "sum"})
    return pd.DataFrame(pd.Series([temp_df.usage.mean(), temp_df.usage.max(), temp_df.usage.min(), temp_df.usage.std(),
                                   temp_df.usage.median()])).transpose()


labels = pd.read_csv("my_data/labels.csv")
good_users = np.array(labels[labels.unknown != 1].img.tolist())
df = load_data_frame()
df = df.groupby("id").apply(vector)
df = df.fillna(0)
df = df.reset_index()
users_id = df.id
df = df.drop(columns=["id"])

x = df.to_numpy()
x = StandardScaler().fit_transform(x)

models = [
    KMeansConstrained(n_clusters=2, size_min=10, n_jobs=-1),
    KMeansConstrained(n_clusters=3, size_min=10, n_jobs=-1),
    KMeansConstrained(n_clusters=4, size_min=10, n_jobs=-1),
    KMeansConstrained(n_clusters=5, size_min=10, n_jobs=-1),
]

best = None
best_value = 0
silhouette_scores = []
for i, model in enumerate(models):
    y_prediction = model.fit_predict(x)

    y_set = set(y_prediction)
    if len(y_set) > 1:
        sil = metrics.silhouette_score(x, y_prediction)
        silhouette_scores.append(sil)
        if best is None or sil > best_value:
            best = model
            best_value = sil
    else:
        print("one cluster")

y_prediction = best.predict(x)
y_set = set(y_prediction)
df = load_data_frame()
df = df[df.id.isin(good_users)]
users_clusters = []
for cluster in y_set:
    ss = users_id[y_prediction == cluster]
    print(len(ss))
    users_clusters.append(ss)
    print("cluster {} count = {}".format(cluster, (y_prediction == cluster).sum()))
print(best_value)

with open('user_clusters.pkl', 'wb') as f:
    pickle.dump(users_clusters, f)
