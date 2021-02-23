import pathlib

import pandas as pd

files_path = pathlib.Path("data/")
files = list(files_path.glob("*.xlsx"))
print("files count = {}".format(len(files)))

final_columns = ["id", "usage", "date"]
all_data = pd.DataFrame()

count = 1
for file in files:
    print("reading file {}".format(count))
    data_frame = pd.read_excel(str(file))
    small_data_frame = data_frame[['Customer No.', 'Active energy(+) total(kWh)', 'Gregorian Calendar']]
    small_data_frame = small_data_frame.rename(columns={'Customer No.': "id",
                                                        'Active energy(+) total(kWh)': "usage",
                                                        'Gregorian Calendar': "date"})
    small_data_frame['date'] = pd.to_datetime(small_data_frame['date'])
    all_data = pd.concat([all_data, small_data_frame])
    count += 1

print(all_data.isnull().sum())

all_data = all_data.set_index('date')
all_data = all_data.groupby(all_data['id']).resample('60T').agg({"usage": "max"})  # to 1 hour sampling rate
all_data = all_data.reset_index(level='id', drop=False)

# todo we need data cleaning
# all_data["usage"] = all_data.groupby(all_data['id']).transform(lambda x: x.fillna(x.mean()))
# print(all_data.isnull().sum())

# all_data = all_data.groupby(all_data['id']).resample('D').aggregate({'usage': lambda x: x.tolist()})  # to 1 day list
# all_data = all_data.reset_index()

print("saving")

all_data.to_csv("my_data/all_data.csv")
