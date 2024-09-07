import pandas as pd
from glob import glob

single_file_acc=pd.read_csv("../../data/raw/WorkoutCounter/Arnold-press_Erwin_Heavy_1_2024-09-07T165520.758_acceleration.csv")

single_file_gyr=pd.read_csv("../../data/raw/WorkoutCounter/Arnold-press_Erwin_Heavy_1_2024-09-07T165520.758_angular_velocity.csv")


files=glob("..\\..\\data\\raw\\WorkoutCounter\\*.csv")

len(files) 




data_path="..\\..\\data\\raw\\WorkoutCounter\\"
f=files[0]
print(f)

participant=f.split("-")[0].replace(data_path,"")
print(participant)
label=f.split("-")[1]
category=f.split("-")[2].rstrip("123")

df=pd.read_csv(f)

df["participant"]=participant
df["category"]=category
df["label"]=label





acc_df=pd.DataFrame()
gyr_df=pd.DataFrame()

acc_set=1
gyr_set=1


for f in files:
    data_path="..\\..\\data\\raw\\WorkoutCounter\\"
    participant=f.split("-")[0].replace(data_path,"")
    label=f.split("-")[1]
    category=f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
        
    df=pd.read_csv(f)

    df["participant"]=participant
    df["category"]=category
    df["label"]=label

    if "Accelerometer" in f:
        df["set"]=acc_set
        acc_set+=1
        acc_df=pd.concat([acc_df,df])   

    if "Gyroscope" in f:
        df["set"]=gyr_set
        gyr_set+=1
        gyr_df=pd.concat([gyr_df,df])   

acc_df.head()




acc_df.info()

pd.to_datetime(df["epoch (ms)"], unit="ms")

acc_df.index=pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index=pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")


del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]




files=glob("..\\..\\data\\raw\\WorkoutCounter\\*.csv")

def read_data_from_files(files):
    data_path="..\\..\\data\\raw\\WorkoutCounter
\\"

    acc_df=pd.DataFrame()
    gyr_df=pd.DataFrame()

    acc_set=1
    gyr_set=1


    for f in files:
        participant=f.split("-")[0].replace(data_path,"")
        label=f.split("-")[1]
        category=f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
            
        df=pd.read_csv(f)

        df["participant"]=participant
        df["category"]=category
        df["label"]=label

        if "Accelerometer" in f:
            df["set"]=acc_set
            acc_set+=1
            acc_df=pd.concat([acc_df,df])   

        if "Gyroscope" in f:
            df["set"]=gyr_set
            gyr_set+=1
            gyr_df=pd.concat([gyr_df,df])   

    pd.to_datetime(df["epoch (ms)"], unit="ms")

    acc_df.index=pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index=pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")


    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)


data_merged=pd.concat([acc_df.iloc[:,:3],gyr_df], axis=1)



data_merged.columns=[
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "category",
    "label",
    "set"
]




sampling={
    "acc_x":"mean",
    "acc_y":"mean",
    "acc_z":"mean",
    "gyr_x":"mean",
    "gyr_y":"mean",
    "gyr_z":"mean",
    "label":"last",
    "category":"last",
    "participant":"last",
    "set":"last"
}

days=[g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

data_resampled=pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

data_resampled.info()



data_resampled.to_pickle("..\\..\\data\\interim\\01_data_processed.pkl")
data_resampled.to_csv("..\\..\\data\\interim\\01_data_processed.csv")
