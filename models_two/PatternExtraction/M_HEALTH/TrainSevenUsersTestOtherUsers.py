from multiprocessing import cpu_count, Pool

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

swifter_config = swifter.config

df = pd.read_csv("../../../my_data/MHEALTHDATASET/health.csv")

segment_length = 200
distance_threshold = 0.9
confidence_threshold = 0.75


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

        small_df["segment"] = small_df["order"].swifter.progress_bar(False).apply(lambda x: x // segment_length)
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
    vectors = temp_vec[0][0]
    labels = temp_vec[0][1]

    vectors, labels = sklearn.utils.shuffle(vectors, labels)
    scalers = {}

    for i in range(vectors.shape[2]):
        scalers[i] = MinMaxScaler()
        vectors[:, :, i] = scalers[i].fit_transform(vectors[:, :, i])

    return vectors, labels


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


def find_valuable_clusters(y_pred, temp_train_y, temp_train_x, temp_feature_index):
    def inner(inner_df: pd.DataFrame):
        inner_df.columns = ["value", "label"]
        return inner_df.groupby("label").count().transpose()

    temp_result_df = pd.DataFrame(np.stack([y_pred, temp_train_y], axis=1), columns=["cluster", "label"])
    temp_result_df = apply_parallel(temp_result_df.groupby("cluster"), inner)
    temp_result_df = temp_result_df.reset_index(level=0, drop=True)

    temp_sum = temp_result_df.swifter.progress_bar(False).apply(lambda row: row.sum(), axis=1)
    temp_result_df = temp_result_df.swifter.progress_bar(False).apply(lambda row: row / row.sum(), axis=1)
    temp_result_df = temp_result_df.swifter.progress_bar(False).apply(lambda row: pd.Series([row.max(), row.idxmax()]),
                                                                      axis=1)
    temp_result_df.columns = ["confidence", "label"]
    temp_result_df["sum"] = temp_sum
    temp_result_df["feature"] = temp_feature_index

    clf = NearestCentroid()
    clf.fit(temp_train_x, y_pred)
    temp_result_df["centroids"] = clf.centroids_.tolist()

    temp_result_df = temp_result_df[temp_result_df["confidence"] > confidence_threshold]
    return temp_result_df


def predict_test(temp_pattern_df: pd.DataFrame, temp_test_x, temp_test_y, temp_feature_index):
    single_feature_patterns = temp_pattern_df[temp_pattern_df["feature"] == temp_feature_index]
    temp_distance_matrix = calculate_distance_to_test(temp_test_x, np.array(
        single_feature_patterns["centroids"].tolist()))
    temp_min_distance = np.min(temp_distance_matrix, axis=1)
    temp_min_distance_index = np.argmin(temp_distance_matrix, axis=1)

    temp_predicted_labels = []
    for temp_cluster in temp_min_distance_index:
        try:
            temp_predicted_labels.append(single_feature_patterns.iloc[temp_cluster]["label"])
        except:
            temp_predicted_labels.append(np.nan)
    temp_predicted_labels = np.array(temp_predicted_labels)

    all_prediction_result = pd.Series(temp_predicted_labels, index=range(temp_predicted_labels.shape[0]))
    good_indexes = ~np.isnan(temp_predicted_labels)
    temp_predicted_labels = temp_predicted_labels[good_indexes]
    temp_test_y = temp_test_y.squeeze()[good_indexes]
    min_distance = temp_min_distance[good_indexes]
    temp_predicted_labels = temp_predicted_labels[min_distance < distance_threshold] == temp_test_y[
        min_distance < distance_threshold]

    return all_prediction_result, temp_predicted_labels


def vote_label(row):
    temp_row = row.dropna()
    if temp_row.shape[0] == 0:
        return -1
    else:
        return np.argmax(np.bincount(temp_row.astype(np.int)))


def train(temp_df: pd.DataFrame):
    temp_vectors = temp_df.groupby("id").apply(vectorization).reset_index(drop=True)
    train_x, train_y = shuffle_split_scale(temp_vectors)

    temp_pattern_df = pd.DataFrame()
    for feature_index in range(train_x.shape[2]):
        one_feature_train_x = train_x[:, :, feature_index]
        distance_matrix = calculate_distance_matrix(one_feature_train_x)
        clu = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None, affinity="precomputed",
                                      linkage="average")
        clu = clu.fit(distance_matrix)
        result_df = find_valuable_clusters(clu.labels_, train_y.squeeze(), one_feature_train_x, feature_index)
        temp_pattern_df = temp_pattern_df.append(result_df)

    return temp_pattern_df


def test(temp_patterns_df: pd.DataFrame, temp_df: pd.DataFrame):
    temp_vectors = temp_df.groupby("id").apply(vectorization).reset_index(drop=True)
    test_x, test_y = shuffle_split_scale(temp_vectors)

    temp_prediction_df = pd.DataFrame()
    for feature_index in range(test_x.shape[2]):
        one_feature_test_x = test_x[:, :, feature_index]
        result_column, predicted_labels = predict_test(temp_patterns_df, one_feature_test_x, test_y, feature_index)
        temp_prediction_df["feature_{}".format(feature_index)] = result_column
        print("feature {} test accuracy: ", predicted_labels.sum() / predicted_labels.shape[0])

    final_label = temp_prediction_df.apply(vote_label, axis=1)
    temp = final_label.to_numpy() == test_y.squeeze()
    print("total test accuracy: ", temp.sum() / temp.shape[0])
    final_label = pd.DataFrame([test_y.squeeze(), final_label]).transpose()
    final_label.columns = ["label", "predict"]
    return final_label


train_users_id, test_users_id = train_test_split(df.id.unique(), test_size=0.3, random_state=42)

pattern_df = pd.DataFrame()
print("total users count: ", df.id.unique().shape[0])
for user_id in train_users_id:
    print("extracting patterns on user ", user_id)
    user_df = df[df.id == user_id]
    pattern_df = pattern_df.append(train(user_df))

pattern_df.to_csv("../../my_data/MHEALTHDATASET/patterns.csv")

prediction_df = pd.DataFrame()
for user_id in test_users_id:
    user_df = df[df.id == user_id]
    prediction_df = prediction_df.append(test(pattern_df, user_df))

prediction_df.to_csv("../../my_data/MHEALTHDATASET/prediction.csv")