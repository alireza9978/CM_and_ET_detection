import pandas as pd
import numpy as np


def read_dataset(path):
    return pd.read_csv(path)


dataset = read_dataset('H:/Projects/Datasets/irish.csv')
dataset['date'] = pd.to_datetime(dataset.date)
dataset['hours'] = dataset.date.dt.hour
dataset['date'] = dataset['date'].dt.strftime('%Y-%m-%d')
grid = dataset.reset_index().groupby(['id', 'date', 'hour'])['usage'].aggregate('first').unstack()
print(grid)
