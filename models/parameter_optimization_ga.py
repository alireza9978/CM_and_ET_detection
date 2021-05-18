import pickle

import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga

from models.Preprocessing import load_data_frame
from models.detection import set_detection_parameters, detect
from models.fill_nan import FillNanMode
from models.filters import select_one_user

with open('user_clusters.pkl', 'rb') as f:
    users_clusters = pickle.load(f)
labels = pd.read_csv("my_data/labels.csv")

df = load_data_frame("../my_data/all_data.csv", False, True, FillNanMode.drop)
good_users = np.array(labels[labels.unknown != 1].img.tolist())
df = df[df.id.isin(good_users)]

varbound = np.array([[1, 7], [10, 50], [1, 5], [0, 30], [0.5, 1]])
vartype = np.array([['int'], ['int'], ['real'], ['int'], ["real"]])
algorithm_param = {'max_num_iteration': 50,
                   'population_size': 50,
                   'mutation_probability': 0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type': 'uniform',
                   'max_iteration_without_improv': None}

print("starting")

for temp_users_id in users_clusters:
    temp_user_df = df[df.id.isin(temp_users_id)]


    def accuracy(a):
        set_detection_parameters(a[0], a[1], a[2], a[3], a[4])
        global temp_user_df
        temp_detection = detect(temp_user_df)
        train_x = temp_user_df.id.unique()
        correct_count = 0
        for user_id in train_x:
            temp_user = select_one_user(temp_detection, user_id)
            user_label = labels[labels.img == user_id].all()
            is_mining = temp_user.mining.sum() > 0
            is_theft = temp_user.theft.sum() > 0
            if user_label.normal:
                if not is_mining or not is_theft:
                    correct_count += 1
            elif user_label.mining:
                if is_mining:
                    correct_count += 1
            elif user_label.theft:
                if is_theft:
                    correct_count += 1
            elif is_theft or is_mining:
                correct_count += 1
        return 1 - (correct_count / len(train_x))


    model = ga(function=accuracy, dimension=5, variable_type_mixed=vartype, variable_boundaries=varbound,
               function_timeout=1000, algorithm_parameters=algorithm_param, convergence_curve=False)
    model.run()
    print(model.best_variable)
    print(accuracy(model.best_variable))
