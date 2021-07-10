from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode
from models.filters import *
from models.detection import *
from mashhad_test.Visualization import plot
from models.visualization import plot_detection

if __name__ == '__main__':
    temp_df = load_data_frame("my_data/mashhad_withNan.csv", False, False, FillNanMode.linear_auto_fill)
    temp_df = usage_mean_above_input_percent(temp_df, 0.8, "1D", 50)
    temp_df = usage_mean_above_input_percent(temp_df, 0.8, "7D", 400)
    temp_df.set_index("date").groupby("id").apply(plot)
    model = Detection("7D", 20, 2.5, 10, 8)
    detection = model.detect(temp_df)
    print(detection.shape)
    print(len(detection.id.unique()))
    for user_id in detection.id.unique():
        plot_detection(detection, user_id, "my_figures/mashhad_mining/{}.jpeg".format(user_id), mining=True)
