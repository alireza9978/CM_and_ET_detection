import os

import arabic_reshaper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bidi import algorithm as bidialg
from matplotlib import font_manager as fm
from persiantools.jdatetime import JalaliDateTime

from models.filters import select_one_user


# function to plot output of detection method
# if input data frame dose not contain anomaly method will not plot any thing
def plot_detection(temp_df: pd.DataFrame, temp_user_id: int, fig_name: str, mining: bool = False, theft: bool = False):
    temp_df = select_one_user(temp_df, temp_user_id)

    if not (temp_df["mining"].sum() > 0 and mining) and not (temp_df["theft"].sum() > 0 and theft):
        return

    fig, axe = plt.subplots(1, 1, figsize=(10, 5))

    font_path = os.path.join("../fonts/bnazanin.TTF")
    axes_prop = fm.FontProperties(fname=font_path, size=12)
    prop = fm.FontProperties(fname=font_path, size=16)

    indexes = temp_df.index.map(JalaliDateTime).map(lambda x: x.strftime("%Y/%m/%d"))
    index_count = len(indexes)
    axe.plot(indexes, temp_df["usage"], 'black', label=bidialg.get_display(arabic_reshaper.reshape(u"مصرف")))

    if temp_df["mining"].sum() > 0 and mining:
        indexes = temp_df.loc[temp_df["mining"]].index.map(JalaliDateTime).map(lambda x: x.strftime("%Y/%m/%d"))
        axe.plot(indexes, temp_df.loc[temp_df["mining"], "usage"], 'y',
                 label=bidialg.get_display(arabic_reshaper.reshape(u"استخراج رمز ارز")), marker="x", markersize=5)
    if temp_df["theft"].sum() > 0 and theft:
        indexes = temp_df.loc[temp_df["theft"]].index.map(JalaliDateTime).map(lambda x: x.strftime("%Y/%m/%d"))
        axe.plot(indexes, temp_df.loc[temp_df["theft"], "usage"], 'r', marker="x", markersize=5,
                 label=bidialg.get_display(arabic_reshaper.reshape(u"برق دزدی")))
    axe.set_xticks(np.arange(0, index_count, index_count // 25))
    axe.legend(prop=prop)
    axe.set_ylabel(bidialg.get_display(arabic_reshaper.reshape(u"مصرف به کیلووات ساعت")), fontproperties=prop)
    axe.set_xlabel(bidialg.get_display(arabic_reshaper.reshape(u"زمان")), fontproperties=prop)

    for label in axe.get_yticklabels():
        label.set_fontproperties(axes_prop)

    for label in axe.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')
        label.set_fontproperties(axes_prop)
    plt.title(' {} '.format(temp_user_id) + bidialg.get_display(arabic_reshaper.reshape(u"کاربر با شناسه")),
              fontproperties=prop)
    fig.tight_layout()
    plt.savefig(fig_name)
    plt.close()
