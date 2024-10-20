import numpy as np
import onnxruntime as rt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle("../data/interim/03_data_features.pkl")


# ---------------------------a-----------------------------------
# Create a training and test set
# --------------------------------------------------------------

df_train = df.drop(["participant", "category", "set"], axis=1)

X = df_train.drop("label", axis=1)
Y = df_train["label"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42, stratify=Y
)

X_train.info()


sess=rt.InferenceSession("../models/neuralNetwork_model_6_features.onnx", providers=rt.get_available_providers)

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[1].name

pred_onx=sess.run(
        [label_name], {input_name: X_test.astype(np.float32)})[0]

print(pred_onx)