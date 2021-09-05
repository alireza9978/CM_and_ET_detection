import pandas as pd

df = pd.read_csv("../../my_data/MHEALTHDATASET/health.csv")

segment_count = 4000

def vectorization(temp_df: pd.DataFrame):
    size = temp_df.shape[0] // segment_count

    temp_df["segment"] = temp_df["order"] // size
    good_segment = temp_df[['id', 'segment']].groupby("segment").count() == size
    temp_df = temp_df[temp_df.segment.isin(good_segment[good_segment["id"]].index)]


vec = df.groupby("id").apply(vectorization)