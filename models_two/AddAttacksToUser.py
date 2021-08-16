import random

import numpy as np
import pandas as pd


def read_user():
    df = pd.read_csv('/content/gdrive/MyDrive/datasets/daily-electricity-usage/irish_grid.csv', delimiter=',')

    # plotting 1 week for 1 user
    user_id = 1000
    user = df.loc[(df['id'] == user_id)]
    user.drop(columns=['date', 'id'], inplace=True)
    print(user)
    return user
    # for i in range(7):
    #   plt.plot(user.iloc[i])


# Attacks

# random 0.2 - 0.8
def attack1(seg):
    return seg * random.uniform(0.2, 0.8)


# random each element 0.2 - 0.8
def attack2(seg):
    return [i * random.uniform(0.2, 0.8) for i in seg]


# mean
def attack3(seg):
    return np.full(seg.shape, seg.mean())


# fraction of mean
def attack4(seg):
    return np.full(seg.shape, seg.mean() * random.uniform(0.2, 0.8))


# reverse
def attack5(seg):
    return seg[::-1]


# zero
def attack6(seg):
    return np.zeros(seg.shape)


def attack7(seg):
    pass


def attack_generator(split_df, split_size):
    MIN_MALICIOUS_HOURS = 48
    MAX_MALICIOUS_HOURS = split_size * 24

    random_start = random.randint(0,
                                  MAX_MALICIOUS_HOURS - MIN_MALICIOUS_HOURS)  # choose a random starting point for
    # anomaly in all the hours with 48 hours minimum
    affected = random.randint(MIN_MALICIOUS_HOURS,
                              MAX_MALICIOUS_HOURS - random_start)  # choose number of affected hours
    attack_list = [attack1, attack2, attack3, attack4, attack5, attack6]
    # print('affected = {}, random_start = {}'.format(affected,random_start))
    if random.random() < 0.3:
        labels.append(1)
        split_df[random_start:random_start + affected] = attack_list[random.randint(0, len(attack_list) - 1)](
            split_df[random_start:random_start + affected])
    else:
        labels.append(0)
    return split_df


# We will create 2 datasets. In the first one we have daily vectors and a labels for each 7 days.
# In the second dataset we have weekly vectors and one label for each row

# Add anomalies to user's data


# We split the user's data into segments with pre-defined length
# Then we randomly choose a number of them to change
if __name__ == '__main__':
    SPLIT_SIZE = 7
    user = read_user()
    seg_num = len(user) // SPLIT_SIZE

    labels = []
    split_df_list = [user.iloc[i:i + SPLIT_SIZE].to_numpy().flatten() for i in
                     range(0, len(user) - SPLIT_SIZE + 1, SPLIT_SIZE)]
    changed_split_df_list = [attack_generator(i, SPLIT_SIZE) for i in split_df_list]
    labels = np.array(labels)
    # dataset in the weekly form
    new_user = pd.DataFrame(changed_split_df_list)
    new_user.to_csv('/content/gdrive/MyDrive/datasets/daily-electricity-usage/user1000_week.csv')
    # dataset in the daily form
    changed_split_df_list = [i.reshape(-1, 24) for i in changed_split_df_list]
    changed_split_df_array = np.concatenate(changed_split_df_list, axis=0)
    new_user = pd.DataFrame(changed_split_df_array)
    new_user.to_csv('/content/gdrive/MyDrive/datasets/daily-electricity-usage/user1000_day.csv')
