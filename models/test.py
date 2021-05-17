from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode

if __name__ == '__main__':
    path = "../sample_data/hourly_sample_accumulative-usage_gregorian-date.csv"
    df = load_data_frame(path, False, True, FillNanMode.without)
    print(df)

    path = "../sample_data/monthly_sample_non-accumulative-usage_gregorian-date.csv"
    df = load_data_frame(path, True, False, FillNanMode.without)
    print(df)
