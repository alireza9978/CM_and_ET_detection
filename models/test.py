from models.Preprocessing import load_data_frame
from models.detection import Detection
from models.fill_nan import FillNanMode
from models.visualization import plot_detection

if __name__ == '__main__':
    path = "../sample_data/hourly_sample_accumulative-usage_gregorian-date.csv"
    hourly_df = load_data_frame(path, False, True, FillNanMode.drop)
    # hourly_df = select_one_user(hourly_df, user_id=132)

    # path = "../sample_data/monthly_sample_non-accumulative-usage_gregorian-date.csv"
    # monthly_df = load_data_frame(path, True, False, FillNanMode.without)

    # hourly_df = select_random_user(hourly_df)
    detection_clf = Detection("7D", 30, 2.5, 14, 8)
    detection = detection_clf.detect(hourly_df)
    print(detection.shape)
    print(len(detection.id.unique()))
    for user_id in detection.id.unique():
        plot_detection(detection, user_id, "../my_figures/mining/{}.jpeg".format(user_id), mining=True)
        plot_detection(detection, user_id, "../my_figures/theft/{}.jpeg".format(user_id), theft=True)
