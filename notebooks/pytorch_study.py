import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import optuna

# Define a simple feedforward neural network using PyTorch
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_units1, hidden_units2, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units1)  # First hidden layer
        self.fc2 = nn.Linear(hidden_units1, hidden_units2)  # Second hidden layer
        self.output = nn.Linear(hidden_units2, num_classes)  # Output layer
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation for first hidden layer
        x = torch.relu(self.fc2(x))  # ReLU activation for second hidden layer
        x = torch.softmax(self.output(x), dim=1)  # Softmax for output layer (classification)
        return x

def objective(trial):
    # Hyperparameters to tune
    hidden_units1 = trial.suggest_int('hidden_units1', 16, 128)
    hidden_units2 = trial.suggest_int('hidden_units2', 16, 128)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    num_epochs = 50  # You can also tune this if needed

    # Load and preprocess data
    df = pd.read_pickle("../data/pipeline/02_outliers_removed_chauvenet.pkl")
    df_train = df.drop(["participant", "category", "set"], axis=1)
    X = df_train.drop("label", axis=1).values
    Y = df_train["label"].values

    # Convert labels to one-hot encoding
    lb = LabelBinarizer()
    Y = lb.fit_transform(Y)

    # Split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=42, stratify=Y
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the model, loss function, and optimizer
    input_size = X_train.shape[1]
    num_classes = Y_train.shape[1]  # Assuming one-hot encoded labels
    model = SimpleNN(input_size=input_size, hidden_units1=hidden_units1, hidden_units2=hidden_units2, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training the model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, torch.max(batch_Y, 1)[1])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Validation loss for pruning
    return running_loss / len(train_loader)

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters found
print("Best hyperparameters: ", study.best_params)

# If you want to visualize the results
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_parallel_coordinate(study)
