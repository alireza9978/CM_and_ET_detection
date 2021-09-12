import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def read_dataset(path):
    return pd.read_csv(path)


dataset = read_dataset('my_data/irish.csv')
# dataset['date'] = pd.to_datetime(dataset.date)
# dataset['hour'] = dataset.date.dt.hour
# dataset['date'] = dataset['date'].dt.strftime('%Y-%m-%d')
scaler_model = MinMaxScaler(feature_range=(0, 255))
# usage = dataset.groupby(['id'])['usage'].apply(lambda x: scaler_model.fit_transform(x.to_numpy().reshape(-1, 1)).squeeze())

# grid = dataset.reset_index().groupby(['id', 'date', 'hour'])['usage'].aggregate('first').unstack()
# grid.to_csv('my_data/irish_grid.csv')

print('hello')
