from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.utils
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import dtw

from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode
from models.filters import data_frame_agg
from models_two.AddAttacksToUser import attack1, attack2, attack3, attack4, attack5, attack6


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def vectorization(temp_df: pd.DataFrame, aggregation_value: str, segment_length: int):
    def vectorization_single_user(inner_df: pd.DataFrame):
        inner_df = inner_df.reset_index()

        inner_df["segment"] = inner_df.index // segment_length
        good_segment = inner_df[['usage', 'segment']].groupby("segment").count() == segment_length
        inner_df = inner_df[inner_df.segment.isin(good_segment[good_segment.usage].index)]

        inner_df = inner_df.usage.to_numpy().reshape(-1, segment_length)
        inner_df = np.round(inner_df, 1)

        return inner_df

    temp_df = data_frame_agg(temp_df, aggregation_value)
    temp_df = temp_df.groupby("id").apply(vectorization_single_user)
    temp_df = temp_df.reset_index()
    temp_df.columns = ["id", "usages"]
    temp_df = np.concatenate(temp_df.usages)
    temp_df = sklearn.utils.shuffle(temp_df)

    return temp_df, None
    # # select random part of data set and apply random attack
    # sampled_index_to_attack = np.random.choice(np.arange(temp_df.shape[0]), int(0.5 * temp_df.shape[0]), replace=False)
    # temp_df_to_attack = temp_df[sampled_index_to_attack]
    # temp_labels = []
    # for i in range(temp_df_to_attack.shape[0]):
    #     attack_number = round(random() * 5)
    #     temp_labels.append(attack_number)
    #     temp_df_to_attack[i] = attacks[attack_number](temp_df_to_attack[i])

    # concatenate anomaly data points with normal data
    # temp_labels = np.concatenate([np.full(temp_df.shape[0], -1), np.array(temp_labels)])
    # temp_df = np.concatenate([temp_df, temp_df_to_attack])
    # temp_df, temp_labels = sklearn.utils.shuffle(temp_df, temp_labels)

    # return temp_df, temp_labels


def calculate_distance_matrix(temp_df):
    def process(i, j):
        return i, j, dtw(temp_df[i], temp_df[j])

    results = Parallel(n_jobs=cpu_count())(
        delayed(process)(i, j) for i in range(temp_df.shape[0]) for j in range(i + 1, temp_df.shape[0]))

    distance_matrix = np.zeros((temp_df.shape[0], temp_df.shape[0]))
    for row in results:
        distance_matrix[row[0]][row[1]] = row[2]
        distance_matrix[row[1]][row[0]] = row[2]

    return distance_matrix


if __name__ == '__main__':

    states = [["1H", 24]]

    # load data set
    path = "/mnt/79e06c5d-876b-45fd-a066-c9aac1a1c932/Dataset/Power Distribution/irish.csv"
    df = load_data_frame(path, False, False, FillNanMode.linear_auto_fill, True)

    # remove some part of data to make sure all of data start from same date
    maximum_start_date = df[["id", "date"]].groupby("id").min().date.unique().max()
    df = df[df.date >= pd.to_datetime(maximum_start_date)]

    # remove data of users that their data points count is less than average
    user_data_points_count = df[["id", "usage"]].groupby("id").count()
    proper_users_ids = (user_data_points_count[user_data_points_count >= user_data_points_count.mean()].dropna()).index
    df = df[df.id.isin(proper_users_ids)]

    # define attacks
    attacks = [attack1, attack2, attack3, attack4, attack5, attack6]

    for state in states:
        # convert user usage to same length vectors
        temp_vec, labels = vectorization(df, state[0], state[1])
        print(temp_vec.shape)

        # select a smaller part of vectors randomly to reduce the calculation
        sampled_index = np.random.choice(np.arange(temp_vec.shape[0]), int(0.1 * temp_vec.shape[0]), replace=False)
        # labels = labels[sampled_index]
        temp_vec = temp_vec[sampled_index]
        print(temp_vec.shape)

        # scale all values to lie between zero and one
        scl = MinMaxScaler()
        temp_vec = scl.fit_transform(temp_vec)

        # cluster usage vectors to find unique clusters
        # clu = DBSCAN(0.2, min_samples=50, metric=dtw)
        # clu = DBSCAN(0.25, min_samples=10)
        # answer = clu.predict(temp_vec)
        # plt.hist(answer)
        # plt.show()

        clu = AgglomerativeClustering(distance_threshold=0.75, n_clusters=None, affinity="precomputed",
                                      linkage="average")
        matrix = calculate_distance_matrix(temp_vec)
        clu = clu.fit(matrix)
        plt.figure(1, (15, 8))
        plt.title('Hierarchical Clustering Dendrogram')
        # plot the top three levels of the dendrogram
        plot_dendrogram(clu, truncate_mode='level', p=5)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()
        plt.close()

        plt.hist(clu.labels_)
        plt.show()
        plt.close()

        print(np.unique(clu.labels_, return_counts=True))

        plt.figure(1, (10, 15))
        centers = []
        for i in range(clu.n_clusters_):
            temp_center = np.mean(temp_vec[clu.labels_ == i], axis=0)
            axe = plt.subplot(clu.n_clusters_, 1, i + 1)
            axe.plot(temp_center)
            centers.append(temp_center)
        plt.show()
        plt.close()

        print(clu)
