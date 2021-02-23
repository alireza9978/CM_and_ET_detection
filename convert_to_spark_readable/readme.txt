main xlsx files are availabel in data folder
change them to hourly data and write in csv with file zero the out put file is all_data.csv
spark code expect file that contains four columns
# row_number, date, id as variable, usage as power
input data is accumulative and in file one we changed all_data.csv to expected format
the final_output is spark_readable.csv
