import pandas as pd
from matplotlib import pyplot as plt
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot

from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode

SEGMENT_LENGTH = 7 * 24


def transform(temp_df: pd.DataFrame):
    user_id = temp_df.id.values[0]

    temp_df = temp_df.reset_index()
    temp_df["segment"] = temp_df.index // SEGMENT_LENGTH
    good_segment = temp_df[['usage', 'segment']].groupby("segment").count() == SEGMENT_LENGTH
    temp_df = temp_df[temp_df.segment.isin(good_segment[good_segment.usage].index)]

    temp_df = temp_df.usage.to_numpy().reshape(-1, SEGMENT_LENGTH)
    for j, transformer in enumerate(transformers):
        name = transformers_name[j]
        images = transformer.transform(temp_df)

        for i, image in enumerate(images):
            plt.imsave("my_figures/{}/{}_{}.jpeg".format(name, user_id, i), image)
            plt.close()

    images = RecurrencePlot(dimension=1, threshold="point", percentage=20).fit_transform(temp_df)
    name = "RecurrencePlot"
    for i, image in enumerate(images):
        plt.imsave("my_figures/{}/{}_{}.jpeg".format(name, user_id, i), image)
        plt.close()


path = "my_data/irish.csv"
df = load_data_frame(path, False, False, FillNanMode.linear_auto_fill, True)
transformers = [GramianAngularField(), MarkovTransitionField()]
transformers_name = ["GramianAngularField", "MarkovTransitionField"]
df.set_index("date").groupby("id").apply(transform)
