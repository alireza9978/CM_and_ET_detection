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
train_df, test_df = get_train_test()

train_df = train_df.groupby("id").apply(vector)
train_df = train_df.fillna(0)
train_df = train_df.reset_index()
train_users_id = train_df.id
train_df = train_df.drop(columns=["id"])

test_df = test_df.groupby("id").apply(vector)
test_df = test_df.fillna(0)
test_df = test_df.reset_index()
test_users_id = test_df.id
test_df = test_df.drop(columns=["id"])

x_train = train_df.to_numpy()
x_test = test_df.to_numpy()
clf = StandardScaler()
x_train = clf.fit_transform(x_train)
x_test = clf.transform(x_test)

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
    y_prediction = model.fit_predict(x_train)

    y_set = set(y_prediction)
    if len(y_set) > 1:
        sil = metrics.silhouette_score(x_train, y_prediction)
        print(sil)
        silhouette_scores.append(sil)
        if best is None or sil > best_value:
            best = model
            best_value = sil
    else:
        print("one cluster")

y_train_prediction = best.predict(x_train)
y_test_prediction = best.predict(x_test)
y_set = set(y_train_prediction)
train_users_clusters = []
test_users_clusters = []
for cluster in y_set:
    print("here")
    temp = train_users_id[y_train_prediction == cluster]
    train_user_cluster_id = temp[temp.isin(good_users)]
    temp = test_users_id[y_test_prediction == cluster]
    test_user_cluster_id = temp[temp.isin(good_users)]
    train_users_clusters.append(train_user_cluster_id)
    test_users_clusters.append(test_user_cluster_id)
    print("train_cluster {} count = {}".format(cluster, train_user_cluster_id.shape))
    print("test_cluster {} count = {}".format(cluster, test_user_cluster_id.shape))
print(best_value)

with open('train_user_clusters.pkl', 'wb') as f:
    pickle.dump(train_users_clusters, f)

with open('test_user_clusters.pkl', 'wb') as f:
    pickle.dump(test_users_clusters, f)
