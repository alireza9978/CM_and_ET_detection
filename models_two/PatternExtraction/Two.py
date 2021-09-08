import numpy as np
import pandas
import pandas as pd
import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("../../my_data/MHEALTHDATASET/health.csv")

segment_length = 100


def vectorization(temp_df: pd.DataFrame):
    temp_df = temp_df.reset_index(drop=True)
    temp_vectors = np.ndarray(shape=(0, segment_length, 23))
    labels_vector = np.ndarray(shape=(0, 1))
    start = 0
    end = start + 1
    max_end = temp_df.shape[0]
    while True:
        last_label = temp_df.label[start]
        while temp_df.label[end] == last_label:
            end += 1
            if end >= max_end:
                break

        if last_label == 0:
            start = end
            end += 1
            if end >= max_end:
                break
            continue
        small_df = temp_df[start:end]

        small_df["segment"] = small_df["order"] // segment_length
        good_segment = small_df[['id', 'segment']].groupby("segment").count() == segment_length
        small_df = small_df[small_df.segment.isin(good_segment[good_segment["id"]].index)]
        temp_vec = small_df.drop(columns=["label", "segment", "id", "order"]).to_numpy().reshape(-1, segment_length, 23)
        temp_vectors = np.concatenate([temp_vectors, temp_vec])
        labels_vector = np.concatenate([labels_vector, np.full((temp_vec.shape[0], 1), last_label, np.int)])
        start = end
        end += 1
        if end >= max_end:
            break

    return temp_vectors, labels_vector


# to speed up process
df = df[df.id == 1]
vec = df.groupby("id").apply(vectorization)

vectors = vec[1][0]
labels = vec[1][1]
vectors, labels = sklearn.utils.shuffle(vectors, labels)

for i in range(vectors.shape[2]):
    one_feature = vectors[:, :, i]

    # train_x, test_x, train_y, test_y = train_test_split(one_feature, labels, test_size=0.33, random_state=42)

    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(one_feature)
    train_y = labels
    # test_x = scaler.transform(test_x)

    clu = AgglomerativeClustering(distance_threshold=8, n_clusters=None)
    clu = clu.fit(train_x)

    # plt.figure(1, (15, 8))
    # plt.title('Hierarchical Clustering Dendrogram')
    # # plot the top three levels of the dendrogram
    # plot_dendrogram(clu, truncate_mode='level', p=5)
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    # plt.show()
    # plt.close()


    def inner(inner_df: pd.DataFrame):
        inner_df.columns = ["value", "label"]
        return inner_df.groupby("label").count().transpose()

    result_df = pd.DataFrame(np.stack([clu.labels_, train_y.squeeze()], axis=1), columns=["cluster", "label"])
    result_df = result_df.groupby("cluster").apply(inner)
    result_df = result_df.reset_index(level=1, drop=True)
    result_df = result_df.apply(lambda row: row / row.sum(), axis=1)
    result_df = result_df.apply(lambda row: pd.Series([row.max(), row.idxmax()]), axis=1)

    print(result_df[result_df[0] > 0.75])
