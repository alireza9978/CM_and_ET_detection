import numpy as np
import pywt
from matplotlib import pyplot as plt

from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode
from models.filters import select_random_user, data_frame_agg
from models_two.Visualization import plot


def fourier(temp_df):
    plt.plot(temp_df)
    coe_count = int(temp_df.shape[0] * 1)
    fft_coe = np.fft.fft(temp_df.usage, coe_count)
    signal = np.abs(np.fft.ifft(fft_coe, temp_df.shape[0]))
    temp_df.usage = signal

    plt.plot(temp_df)
    plt.show()


path = "/mnt/79e06c5d-876b-45fd-a066-c9aac1a1c932/Dataset/Power Distribution/irish.csv"
df = load_data_frame(path, False, False, FillNanMode.linear_auto_fill, True)
df, user_id = select_random_user(df)
df.set_index("date").groupby("id").apply(plot)
df = data_frame_agg(df, "1D")
df = df.drop(columns=["id"]).set_index("date")
print(user_id)

titles = ['Approximation', 'Detail Coefficients', 'Real Data']
LL = df.usage
print("main", df.shape[0])

wt_count = 4
data = []
for i in range(wt_count):
    wt_coe = pywt.dwt(LL, 'db3')
    LL, HH = wt_coe
    print(len(LL))
    data.append([LL, HH])

fig = plt.figure(figsize=(6, 3 * wt_count))
for i, a in enumerate(data):
    ax = fig.add_subplot(wt_count, 2, (2 * i) + 1)
    ax.plot(a[0])
    ax.set_title(titles[0] + " " + str(i), fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(wt_count, 2, (2 * i) + 2)
    ax.plot(a[1])
    ax.set_title(titles[1] + " " + str(i), fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()
