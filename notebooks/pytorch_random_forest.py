import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelBinarizer
import optuna
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# Load and preprocess data
df = pd.read_pickle("../data/pipeline/02_outliers_removed_chauvenet.pkl")
df_train = df.drop(["participant", "category", "set"], axis=1)
X = df_train.drop("label", axis=1).values
Y = df_train["label"].values

# Convert labels to one-hot encoding
lb = LabelBinarizer()
Y = lb.fit_transform(Y)

for index, label in enumerate(lb.classes_):
    print(f"Index {index} corresponds to label '{label}'")

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=40, stratify=Y
)

# Define the objective function for Optuna
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 1, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )

    model.fit(X_train, np.argmax(Y_train, axis=1))
    Y_pred = model.predict(X_test)

    return classification_report(np.argmax(Y_test, axis=1), Y_pred, output_dict=True)['weighted avg']['f1-score']


# Run Optuna to find the best hyperparameters
study = optuna.create_study(direction='maximize') 
study.optimize(objective, n_trials=50)

# Print the best hyperparameters found
print("Best hyperparameters: ", study.best_params)

# Retrain the model with the best hyperparameters
best_params = study.best_params
final_model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    random_state=42
)
final_model.fit(X_train, np.argmax(Y_train, axis=1))

# Evaluate the final model
final_predictions = final_model.predict(X_test)
print("Scikit-learn model evaluation:")
print(classification_report(np.argmax(Y_test, axis=1), final_predictions, target_names=lb.classes_))

# Convert the trained model to ONNX format
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(final_model, initial_types=initial_type)

# Save the ONNX model
with open("final_model_chav.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Load the ONNX model and make predictions on the test set
onnx_session = ort.InferenceSession("final_model_chav.onnx")
onnx_input_name = onnx_session.get_inputs()[0].name

# Prepare ONNX inputs (Note: X_test should be converted to the correct dtype if needed)
onnx_inputs = {onnx_input_name: X_test.astype(np.float32)}
onnx_predictions = onnx_session.run(None, onnx_inputs)[0]

# ONNX model outputs might be in a different shape (e.g., probabilities). Convert to class indices if necessary.
# Assuming it returns class indices similar to the original model
if onnx_predictions.ndim > 1:
    onnx_predictions = np.argmax(onnx_predictions, axis=1)

# Print ONNX model evaluation
print("ONNX model evaluation:")
print(classification_report(np.argmax(Y_test, axis=1), onnx_predictions, target_names=lb.classes_))

# Compare accuracy
sklearn_accuracy = accuracy_score(np.argmax(Y_test, axis=1), final_predictions)
onnx_accuracy = accuracy_score(np.argmax(Y_test, axis=1), onnx_predictions)

print(f"Accuracy - Scikit-learn model: {sklearn_accuracy}")
print(f"Accuracy - ONNX model: {onnx_accuracy}")
