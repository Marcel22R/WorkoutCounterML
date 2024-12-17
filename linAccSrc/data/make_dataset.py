import pandas as pd
from glob import glob

files = glob(
    "C:\\Users\\Marcel\\Desktop\\Python\\WorkoutCounterML\\data\\raw\\WorkoutCounterLinAcc\\*.csv"
)


def read_data_from_files(files):
    data_path="C:\\Users\\Marcel\\Desktop\\Python\\WorkoutCounterML\\data\\raw\\WorkoutCounterLinAcc\\"
    acc_df=pd.DataFrame()

    acc_set=1

    for f in files:
        participant = f.split("_")[1]
        label = f.split("_")[0].replace(data_path, "")
        category = f.split("_")[2]

        df = pd.read_csv(f)

        df["participant"] = participant
        df["category"] = category
        df["label"] = label

        if "linearAcceleration" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])

    pd.to_datetime(df["Timestamp (ms)"], unit="ms")
    acc_df.index = pd.to_datetime(acc_df["Timestamp (ms)"], unit="ms")

    del acc_df["Timestamp (ms)"]
    del acc_df["ISO DateTime"]
    del acc_df["Elapsed Time (ms)"]

    return acc_df


acc_df=read_data_from_files(files)
data_combined=acc_df


data_combined.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "participant",
    "category",
    "label",
    "set",
]


sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "label": "last",
    "category": "last",
    "participant": "last",
    "set": "last",
}

days = [g for n, g in data_combined.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

data_resampled.info()

data_resampled.to_pickle("../../data/interim/01_linAccData_processed_workoutCounter.pkl")
data_resampled.to_csv("../../data/interim/01_linAccData_processed_workoutCounter.csv")
