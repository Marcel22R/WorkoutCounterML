import pandas as pd
from glob import glob

files = glob(
    "C:\\Users\\Marcel\\Desktop\\Python\\WorkoutCounterML\\data\\raw\\WorkoutCounter\\*.csv"
)


def read_data_from_files(files):
    data_path = "C:\\Users\\Marcel\\Desktop\\Python\\WorkoutCounterML\\data\\raw\\WorkoutCounter\\"

    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        participant = f.split("_")[1]
        label = f.split("_")[0].replace(data_path, "")
        category = f.split("_")[2]

        df = pd.read_csv(f)

        df["participant"] = participant
        df["category"] = category
        df["label"] = label

        if "acceleration" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])

        if "angular_velocity" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    pd.to_datetime(df["Timestamp (ms)"], unit="ms")

    acc_df.index = pd.to_datetime(acc_df["Timestamp (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["Timestamp (ms)"], unit="ms")

    del acc_df["Timestamp (ms)"]
    del acc_df["ISO DateTime"]
    del acc_df["Elapsed Time (ms)"]

    del gyr_df["Timestamp (ms)"]
    del gyr_df["ISO DateTime"]
    del gyr_df["Elapsed Time (ms)"]

    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)


data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)


data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "category",
    "label",
    "set",
]


sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "label": "last",
    "category": "last",
    "participant": "last",
    "set": "last",
}

days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

data_resampled.info()


data_resampled.to_pickle("../../data/interim/01_data_processed_workoutCounter.pkl")
data_resampled.to_csv("../../data/interim/01_data_processed_workoutCounter.csv")
