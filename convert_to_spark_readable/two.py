import matplotlib.pyplot as plt
import pandas as pd


def print_null_info(temp_data_frame):
    print("null count in data set: ")
    print(temp_data_frame.isnull().sum())
    print("data shape = ", temp_data_frame.shape)
    print("null percent in data set = ", temp_data_frame.isnull().sum().sum() / temp_data_frame.shape[0])
    temp = temp_data_frame.usage.isnull().groupby(temp_data_frame.id).sum()
    temp = temp.reset_index(drop=True)
    print(temp)
    print("mean = ", temp.mean())
    print("std = ", temp.std())
    print("shape = ", temp.shape)
    print("without null = ", (temp == 0).sum())
    temp.hist(bins=100)
    plt.show()


def print_user_data(temp_data_frame):
    temp = temp_data_frame.usage.groupby(temp_data_frame.id).count()
    temp = temp.reset_index(drop=True)
    print(temp)
    print("mean = ", temp.mean())
    print("std = ", temp.std())
    print("shape = ", temp.shape)
    temp.hist(bins=100)
    plt.show()


data_frame = pd.read_csv("my_data/all_data.csv", date_parser=["datetime"])
data_frame.date = pd.to_datetime(data_frame.date)
# print_null_info(data_frame)
# print_user_data(data_frame)
# print("all data shape = ", data_frame.shape)
x1 = data_frame.usage.groupby(data_frame.id).count()
x2 = data_frame.usage.isnull().groupby(data_frame.id).sum()
x = pd.DataFrame({"total_count": x1 + x2, "null_count": x2})
# print(x)
# print(x.sum())
# print(type(x))
good = data_frame[data_frame.id.isin(x[(x["total_count"] > x["total_count"].mean()) &
                                       (x["null_count"] < x["null_count"].mean())].index)]
# print(good)
# print(good.shape)
good.to_csv("my_data/good_data.csv")
