import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode
from ssqueezepy import cwt
from ssqueezepy.visuals import imshow


def save_plot_spectrogram(address, t, f, sxx):
    fig = plt.figure(frameon=False)
    fig.gca().set_axis_off()
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    fig.set_size_inches(500, 500)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.pcolormesh(t, f, sxx, shading='gouraud')
    fig.savefig(address, bbox_inches='tight', dpi=1, pad_inches=0)
    plt.close(fig)


def save_plot_scalogram(address, wx, scales):
    fig = plt.figure(frameon=False)
    fig.gca().set_axis_off()
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    fig.set_size_inches(500, 500)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    imshow(wx, yticks=scales, abs=1, show=0, fig=fig)
    fig.savefig(address, bbox_inches='tight', dpi=1, pad_inches=0)
    plt.close(fig)


def transform(temp_df: pd.DataFrame):
    user_id = temp_df.id.values[0]
    usage = temp_df.usage.to_numpy()
    f, t, sxx = signal.spectrogram(usage)
    save_plot_spectrogram("my_figures/Spectrogram/{}.jpeg".format(user_id), t, f, sxx)

    wx, scales = cwt(usage, 'morlet')
    save_plot_scalogram("my_figures/Scalogram/{}.jpeg".format(user_id), wx, scales)


path = "my_data/irish.csv"
df = load_data_frame(path, False, False, FillNanMode.linear_auto_fill, True)
df.set_index("date").groupby("id").apply(transform)
