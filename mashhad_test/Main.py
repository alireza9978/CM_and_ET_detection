import matplotlib.pyplot as plt
import numpy as np
import stumpy

from mashhad_test.CreateGrayScaleImage import make_data_set_single_user
from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode
from models.filters import data_frame_agg
from models_two.AddAnomaly import reduce_usage_randomly

if __name__ == '__main__':
    temp_df = load_data_frame("/mnt/79e06c5d-876b-45fd-a066-c9aac1a1c932/Dataset/Power Distribution/irish.csv", False,
                              False, FillNanMode.without, True)
    temp_users_id = np.random.choice(temp_df.id, 1)
    # temp_users_id = np.array([5010892])
    temp_df = temp_df[temp_df.id == temp_users_id[0]]
    make_data_set_single_user(temp_df)

    temp_df_agg = data_frame_agg(temp_df, "7D")

    x = temp_df_agg[["usage", "date"]]
    x = x.reset_index()
    ts = x.usage.to_numpy().reshape(-1)
    window_size = 4  # Approximately, how many data points might be found in a pattern

    matrix_profile = stumpy.stump(ts, m=window_size)

    plt.plot(ts)
    plt.savefig("my_figures/matrix_profiles/user_data.jpeg")
    plt.close()
    plt.plot(matrix_profile[:, 0])
    plt.savefig("my_figures/matrix_profiles/user_profile.jpeg")
    plt.close()

    anomaly_df = temp_df.set_index("date").groupby("id").apply(reduce_usage_randomly)
    anomaly_df = anomaly_df.reset_index()
    anomaly_df = data_frame_agg(anomaly_df, "7D")

    x = anomaly_df[["usage", "date"]]
    x = x.reset_index()
    ts = x.usage.to_numpy().reshape(-1)
    window_size = 4  # Approximately, how many data points might be found in a pattern

    matrix_profile = stumpy.stump(ts, m=window_size)

    plt.plot(ts)
    plt.savefig("my_figures/matrix_profiles/user_data_anomalous.jpeg")
    plt.close()
    plt.plot(matrix_profile[:, 0])
    plt.savefig("my_figures/matrix_profiles/user_profile_anomalous.jpeg")
    plt.close()
