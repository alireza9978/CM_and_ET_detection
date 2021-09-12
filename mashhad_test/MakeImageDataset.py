from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode
from models.filters import *
from models.detection import *
from mashhad_test.Visualization import plot
from models.visualization import plot_detection
import pandas as pd

if __name__ == '__main__':
    bad_user = pd.read_excel("image_dataset/anomal_user.xlsx")
    bad_user.img = bad_user.img.apply(lambda x: x.split(".")[0])
    bad_user = bad_user[bad_user.bad == 1]
    temp_df = load_data_frame("my_data/mashhad_withNan.csv", False, False, FillNanMode.linear_auto_fill)
    temp_df = temp_df[temp_df.id.isin(bad_user.img)]
    temp_df.set_index("date").groupby("id").apply(plot)

