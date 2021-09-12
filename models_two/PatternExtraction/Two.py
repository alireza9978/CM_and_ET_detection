from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import dtw

from models_two.PatternExtraction.One import calculate_distance_matrix

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


def shuffle_split_scale(temp_vec):
    vectors = temp_vec[1][0]
    labels = temp_vec[1][1]
    vectors, labels = sklearn.utils.shuffle(vectors, labels)

    temp_train_x, temp_test_x, temp_train_y, temp_test_y = train_test_split(vectors, labels, test_size=0.33,
                                                                            random_state=42)
    scalers = {}
    for i in range(temp_train_x.shape[2]):
        scalers[i] = MinMaxScaler()
        temp_train_x[:, :, i] = scalers[i].fit_transform(temp_train_x[:, :, i])

    for i in range(temp_test_x.shape[2]):
        temp_test_x[:, :, i] = scalers[i].transform(temp_test_x[:, :, i])

    return temp_train_x, temp_test_x, temp_train_y, temp_test_y


def find_valuable_clusters(y_pred, temp_train_y):
    def inner(inner_df: pd.DataFrame):
        inner_df.columns = ["value", "label"]
        return inner_df.groupby("label").count().transpose()

    temp_result_df = pd.DataFrame(np.stack([y_pred, temp_train_y], axis=1), columns=["cluster", "label"])
    temp_result_df = temp_result_df.groupby("cluster").apply(inner)
    temp_result_df = temp_result_df.reset_index(level=1, drop=True)
    temp_sum = temp_result_df.apply(lambda row: row.sum(), axis=1)
    temp_result_df = temp_result_df.apply(lambda row: row / row.sum(), axis=1)
    temp_result_df = temp_result_df.apply(lambda row: pd.Series([row.max(), row.idxmax()]), axis=1)
    temp_result_df.columns = ["confidence", "label"]
    temp_result_df["sum"] = temp_sum
    temp_result_df["feature"] = feature_index
    temp_result_df = temp_result_df[temp_result_df["confidence"] > 0.75]
    return temp_result_df


def predict_test(temp_train_x, clustering_label_train_x, temp_test_x):
    clf = NearestCentroid()
    clf.fit(temp_train_x, clustering_label_train_x)
    centroids = clf.centroids_

    temp_distance_matrix = calculate_distance_to_test(temp_test_x, centroids)
    temp_min_distance = np.min(temp_distance_matrix, axis=1)
    temp_min_distance_index = np.argmin(temp_distance_matrix, axis=1)

    temp_predicted_labels = []
    for temp_cluster in temp_min_distance_index:
        try:
            temp_predicted_labels.append(result_df.loc[temp_cluster]["label"])
        except:
            temp_predicted_labels.append(np.nan)
    temp_predicted_labels = np.array(temp_predicted_labels)

    all_prediction_result = pd.Series(temp_predicted_labels, index=range(temp_predicted_labels.shape[0]))
    good_indexes = ~np.isnan(temp_predicted_labels)
    temp_predicted_labels = temp_predicted_labels[good_indexes]
    temp_test_y = test_y.squeeze()[good_indexes]
    min_distance = temp_min_distance[good_indexes]
    temp_predicted_labels = temp_predicted_labels[min_distance < 0.75] == temp_test_y[min_distance < 0.75]

    print("accuracy: ", temp_predicted_labels.sum() / temp_predicted_labels.shape[0],
          " -- count: ", temp_predicted_labels.shape[0], " total count: ", temp_train_x.shape[0])

    return all_prediction_result, temp_predicted_labels


# to speed up process
df = df[df.id == 1]
vec = df.groupby("id").apply(vectorization)

train_x, test_x, train_y, test_y = shuffle_split_scale(vec)

prediction_df = pd.DataFrame()
final_df = pd.DataFrame()
for feature_index in range(train_x.shape[2]):
    one_feature_train_x = train_x[:, :, feature_index]
    one_feature_test_x = test_x[:, :, feature_index]

    distance_matrix = calculate_distance_matrix(one_feature_train_x)

    clu = AgglomerativeClustering(distance_threshold=0.75, n_clusters=None, affinity="precomputed", linkage="average")
    clu = clu.fit(distance_matrix)

    result_df = find_valuable_clusters(clu.labels_, train_y.squeeze())

    result_column, predicted_labels = predict_test(one_feature_train_x, clu.labels_, one_feature_test_x)

    prediction_df["feature_{}".format(feature_index)] = result_column

    final_df = final_df.append(result_df)

final_df.to_csv("../../my_data/MHEALTHDATASET/temp_result_dtw.csv")

final_label = prediction_df.apply(lambda row: np.argmax(np.bincount(row.dropna().astype(np.int))), axis=1)
a = final_label.to_numpy() == test_y.squeeze()
print("total test accuracy: ", a.sum()/a.shape[0])
