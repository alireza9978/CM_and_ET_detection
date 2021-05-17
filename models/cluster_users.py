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

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, '#7eb5fc', marker="o", linewidth=0.8)
ax.plot([3], [silhouette_scores[1]], 'r', marker="o", linewidth=2)
ax.text(2.88, silhouette_scores[1] - 0.01, 'best K', color='r', fontsize=12.5)
ax.set_xticks(range(2, len(silhouette_scores) + 2))
ax.set_xticklabels(["2", "3", "4", "5"])
ax.set_ylabel('silhouette', fontsize=14)
ax.set_xlabel('number of clusters', fontsize=14)
plt.savefig("silhouette_scores.pdf")
