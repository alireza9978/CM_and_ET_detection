from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from tslearn.metrics import dtw
from tslearn.utils import to_time_series_dataset

from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode
from models.filters import data_frame_agg
from models_two.LinkedList import Node, SLinkedList


def clustering(df, users_ids):
    def process(i, j):
        return i, j, dtw(df[i], df[j])

    results = Parallel(n_jobs=cpu_count())(
        delayed(process)(i, j) for i in range(df.shape[0]) for j in range(i + 1, df.shape[0]))

    distance_matrix = np.zeros((df.shape[0], df.shape[0]))
    for row in results:
        distance_matrix[row[0]][row[1]] = row[2]
        distance_matrix[row[1]][row[0]] = row[2]
    distance_matrix[distance_matrix == 0] = np.inf
    print("distance matrix is calculated")

    def find_minimum_distance(matrix):
        index = np.unravel_index(np.argmin(matrix, axis=None), matrix.shape)
        value = matrix[index[0]][index[1]]
        return index, value

    def merge_two_cluster(index):
        minimum = min(index[0], index[1])
        maximum = max(index[0], index[1])
        clusters_list[minimum].add_all(clusters_list[maximum])
        clusters_list.append(clusters_list[minimum])
        clusters_list[maximum] = None
        clusters_list[minimum] = None

    def update_distance_matrix(clusters):
        count = len(clusters)
        matrix = np.zeros((count, count))
        matrix[matrix == 0] = np.inf
        for i in range(len(clusters)):
            if clusters[i]:
                cluster_one_items = clusters[i].list()
                for j in range(i + 1, len(clusters)):
                    if clusters[j]:
                        cluster_two_items = clusters[j].list()
                        total_distance = 0
                        for item in cluster_one_items:
                            for other_item in cluster_two_items:
                                total_distance += distance_matrix[item][other_item]
                        total_distance = total_distance / (len(cluster_two_items) * len(cluster_one_items))
                        matrix[i][j] = total_distance
                        matrix[j][i] = total_distance
        return matrix

    def print_clusters(clusters):
        for cluster in clusters:
            if cluster:
                cluster.list_print()
                print("")

        print()

    def save_figures(clusters, state):
        for i, cluster in enumerate(clusters[df.shape[0]:]):
            if cluster:
                Path("my_figures/irish_clustering_state/{}".format(state)).mkdir(parents=True, exist_ok=True)

                fig, axe = plt.subplots(1, 1, figsize=(10, 5))
                for user in cluster.list():
                    user_id = users_ids[user]
                    axe.plot(date_df[date_df.id == user_id]["usage"], label=str(user_id))

                axe.legend()
                for label in axe.get_xticklabels():
                    label.set_rotation(20)
                    label.set_horizontalalignment('right')
                plt.savefig("my_figures/irish_clustering_state/{}/cluster_{}.jpeg".format(state, i))
                plt.close()

    clusters_list = [SLinkedList(Node(i)) for i in range(df.shape[0])]
    clusters_count = df.shape[0]

    children = []
    distances = []
    counts = []

    for iteration in range(df.shape[0] - 1):
        temp_distance_matrix = update_distance_matrix(clusters_list)
        nearest_cluster, distance = find_minimum_distance(temp_distance_matrix)
        merge_two_cluster(nearest_cluster)
        children.append(nearest_cluster)
        distances.append(distance)
        counts.append(clusters_list[-1].size())
        # print_clusters(clusters_list)
        # if clusters_count > 10:
        #     save_figures(clusters_list, clusters_count)
        clusters_count -= 1

    children = np.array(children)
    distances = np.array(distances)
    counts = np.array(counts)
    # print(counts)

    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    linkage_matrix = np.column_stack([children, distances, counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, truncate_mode='level', p=10)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


if __name__ == '__main__':
    path = "/mnt/79e06c5d-876b-45fd-a066-c9aac1a1c932/Dataset/Power Distribution/irish.csv"
    main_df = load_data_frame(path, False, False, FillNanMode.linear_auto_fill, True)
    main_df = data_frame_agg(main_df, "1D")
    main_users_ids = main_df.id.unique()
    date_df = main_df.set_index("date")
    main_df = to_time_series_dataset(main_df.groupby("id").apply(lambda x: x.usage.to_numpy()).to_numpy())
    main_df = main_df.reshape(main_df.shape[0], main_df.shape[1])
    clustering(main_df, main_users_ids)