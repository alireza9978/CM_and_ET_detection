from models.Preprocessing import load_data_frame
from models.detection import detect, set_detection_parameters
from models.fill_nan import FillNanMode
from models.visualization import plot_detection

if __name__ == '__main__':
    path = "../my_data/all_data.csv"
    hourly_df = load_data_frame(path, False, True, FillNanMode.drop)

    # path = "../sample_data/monthly_sample_non-accumulative-usage_gregorian-date.csv"
    # monthly_df = load_data_frame(path, True, False, FillNanMode.without)

    # hourly_df = select_random_user(hourly_df)
    set_detection_parameters("1D", 30, 2.5, 10, 8)
    detection = detect(hourly_df)
    for user_id in detection.id.unique():
        plot_detection(detection, user_id, "../my_figures/mining/{}.jpeg".format(user_id), mining=True)
        plot_detection(detection, user_id, "../my_figures/theft/{}.jpeg".format(user_id), theft=True)
