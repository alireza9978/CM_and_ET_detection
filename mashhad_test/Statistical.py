from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode
import matplotlib.pyplot as plt
from mashhad_test.Visualization import plot

if __name__ == '__main__':
    temp_df = load_data_frame("my_data/mashhad_withNan.csv", False, False, FillNanMode.from_previous_data)
    usage_mean_over_time = temp_df[["date", "usage"]].groupby("date").mean()
    usage_mean_over_time.plot()
    plt.show()
