from multiprocessing import cpu_count, Pool

import time
import numpy as np
import pandas as pd
import sklearn
import swifter
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import dtw

from models_two.PatternExtraction.One import calculate_distance_matrix

a = swifter.config

df = pd.read_csv("/home/alireza/projects/spark_two/my_data/Daily and Sports Activities Data Set/health.csv")

segment_length = 125
feature_count = 45


def vectorization(temp_df: pd.DataFrame):
    temp_df = temp_df.reset_index(drop=True)
    temp_label = temp_df['label'].values[0]

    temp_df["segment"] = temp_df["order"].swifter.apply(lambda x: x // segment_length)
    good_segment = temp_df[['id', 'segment']].groupby("segment").count() == segment_length
    temp_df = temp_df[temp_df.segment.isin(good_segment[good_segment["id"]].index)]
    temp_vectors = temp_df.drop(columns=["label", "segment", "id", "order"]).to_numpy().reshape(-1, segment_length,
                                                                                                feature_count)
    labels_vector = np.full((temp_vectors.shape[0], 1), temp_label, np.int)

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
    vectors = np.ndarray((0, segment_length, feature_count))
    labels = np.ndarray((0, 1))
    for section in temp_vec:
        vectors = np.concatenate([vectors, section[0]])
        labels = np.concatenate([labels, section[1]])

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


def apply_parallel(df_grouped, func):
    result_list = Parallel(n_jobs=cpu_count())(delayed(func)(group) for name, group in df_grouped)
    return pd.concat(result_list)


def parallelize_dataframe(temp_df, func):
    df_split = np.array_split(temp_df, cpu_count())
    pool = Pool(cpu_count())
    temp_df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return temp_df


def find_valuable_clusters(y_pred, temp_train_y):
    def inner(inner_df: pd.DataFrame):
        inner_df.columns = ["value", "label"]
        return inner_df.groupby("label").count().transpose()

    temp_result_df = pd.DataFrame(np.stack([y_pred, temp_train_y], axis=1), columns=["cluster", "label"])
    temp_result_df = apply_parallel(temp_result_df.groupby("cluster"), inner)
    temp_result_df = temp_result_df.reset_index(level=0, drop=True)

    temp_sum = temp_result_df.swifter.apply(lambda row: row.sum(), axis=1)
    temp_result_df = temp_result_df.swifter.apply(lambda row: row / row.sum(), axis=1)
    temp_result_df = temp_result_df.swifter.apply(lambda row: pd.Series([row.max(), row.idxmax()]), axis=1)
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

    if temp_predicted_labels.shape[0] != 0:
        print("accuracy: ", temp_predicted_labels.sum() / temp_predicted_labels.shape[0],
              " -- count: ", temp_predicted_labels.shape[0], " total count: ", temp_test_x.shape[0])
    else:
        print("useless feature")

    return all_prediction_result, temp_predicted_labels


def vote_label(row):
    temp_row = row.dropna()
    if temp_row.shape[0] == 0:
        return -1
    else:
        return np.argmax(np.bincount(temp_row.astype(np.int)))


print("total users count: ", df.id.unique().shape[0])
for user_id in df.id.unique():
    user_df = df[df.id == user_id]
    start_time = time.time()
    vec = df.groupby(["id", "label"]).apply(vectorization).reset_index(drop=True)
    print("vectorization", " --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    train_x, test_x, train_y, test_y = shuffle_split_scale(vec)
    print("shuffle_split_scale", " --- %s seconds ---" % (time.time() - start_time))

    prediction_df = pd.DataFrame()
    final_df = pd.DataFrame()
    for feature_index in range(train_x.shape[2]):
        print("feature number {}".format(feature_index))
        start_time = time.time()
        one_feature_train_x = train_x[:, :, feature_index]
        one_feature_test_x = test_x[:, :, feature_index]
        print("one_feature", " --- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        distance_matrix = calculate_distance_matrix(one_feature_train_x)
        print("calculate_distance_matrix", " --- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        clu = AgglomerativeClustering(distance_threshold=0.75, n_clusters=None, affinity="precomputed", linkage="average")
        clu = clu.fit(distance_matrix)
        print("AgglomerativeClustering", " --- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        result_df = find_valuable_clusters(clu.labels_, train_y.squeeze())
        print("find_valuable_clusters", " --- %s seconds ---" % (time.time() - start_time))

        print("good cluster: ", result_df.shape[0])

        start_time = time.time()
        result_column, predicted_labels = predict_test(one_feature_train_x, clu.labels_, one_feature_test_x)
        print("predict_test", " --- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        prediction_df["feature_{}".format(feature_index)] = result_column
        final_df = final_df.append(result_df)
        print("final_df", " --- %s seconds ---" % (time.time() - start_time))
        print()
        print()

    final_df.to_csv("../../my_data/MHEALTHDATASET/temp_result_dtw.csv")

    final_label = prediction_df.apply(vote_label, axis=1)
    a = final_label.to_numpy() == test_y.squeeze()
    print()
    print("total test accuracy: ", a.sum() / a.shape[0])
