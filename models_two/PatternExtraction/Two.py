import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestCentroid
from models_two.PatternExtraction.One import calculate_distance_matrix
from tslearn.metrics import dtw
from joblib import Parallel, delayed
from multiprocessing import cpu_count

df = pd.read_csv("../../my_data/MHEALTHDATASET/health.csv")

segment_length = 200


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


def calculate_distance_to_test(x, centers):
    def process(i, j):
        return i, j, dtw(x[i], centers[j])

    results = Parallel(n_jobs=cpu_count())(
        delayed(process)(i, j) for i in range(x.shape[0]) for j in range(centers.shape[0]))

    temp_distance_matrix = np.zeros((x.shape[0], centers.shape[0]))
    for row in results:
        temp_distance_matrix[row[0]][row[1]] = row[2]

    return temp_distance_matrix


# to speed up process
df = df[df.id == 1]
vec = df.groupby("id").apply(vectorization)

vectors = vec[1][0]
labels = vec[1][1]
vectors, labels = sklearn.utils.shuffle(vectors, labels)

final_df = pd.DataFrame()
for feature_index in range(vectors.shape[2]):
    one_feature = vectors[:, :, feature_index]

    train_x, test_x, train_y, test_y = train_test_split(one_feature, labels, test_size=0.33, random_state=42)

    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    distance_matrix = calculate_distance_matrix(train_x)

    # clu = AgglomerativeClustering(distance_threshold=5, n_clusters=None)
    # clu = clu.fit(train_x)
    clu = AgglomerativeClustering(distance_threshold=0.75, n_clusters=None, affinity="precomputed", linkage="average")
    clu = clu.fit(distance_matrix)

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
    temp_sum = result_df.apply(lambda row: row.sum(), axis=1)
    result_df = result_df.apply(lambda row: row / row.sum(), axis=1)
    result_df = result_df.apply(lambda row: pd.Series([row.max(), row.idxmax()]), axis=1)
    result_df.columns = ["confidence", "label"]
    result_df["sum"] = temp_sum
    result_df["feature"] = feature_index
    result_df = result_df[result_df["confidence"] > 0.75]

    clf = NearestCentroid()
    clf.fit(train_x, clu.labels_)
    centroids = clf.centroids_

    distance_matrix = calculate_distance_to_test(test_x, centroids)
    min_distance = np.min(distance_matrix, axis=1)
    min_distance_index = np.argmin(distance_matrix, axis=1)

    predicted_labels = []
    for temp_cluster in min_distance_index:
        try:
            predicted_labels.append(result_df.loc[temp_cluster]["label"])
        except:
            predicted_labels.append(np.nan)
    predicted_labels = np.array(predicted_labels)

    good_indexes = ~np.isnan(predicted_labels)
    predicted_labels = predicted_labels[good_indexes]
    test_y = test_y.squeeze()[good_indexes]
    min_distance = min_distance[good_indexes]
    result = predicted_labels[min_distance < 0.75] == test_y[min_distance < 0.75]

    print("accuracy: ", result.sum() / result.shape[0], " -- count: ", result.shape[0], " total count: ", test_x.shape[0])

    final_df = final_df.append(result_df)


final_df.to_csv("../../my_data/MHEALTHDATASET/temp_result_dtw.csv")
