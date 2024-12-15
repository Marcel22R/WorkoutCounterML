import os
import pandas as pd
import matplotlib.pyplot as plt

# Base directory containing the workout data files
base_directory = os.path.join("C:\\", "Users", "Marcel", "Desktop", "Python", "WorkoutCounterML", "data", "raw", "WorkoutCounterLinAcc")

# Iterate through all CSV files in the directory
for file_name in os.listdir(base_directory):
    if file_name.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(base_directory, file_name)
        
        # Load data into a DataFrame
        data = pd.read_csv(file_path)
        
        # Check if the required columns are in the data
        if {'Elapsed Time (ms)', 'Acceleration X', 'Acceleration Y', 'Acceleration Z'}.issubset(data.columns):
            # Plot acceleration data
            plt.figure(figsize=(10, 6))
            plt.plot(data['Elapsed Time (ms)'], data['Acceleration X'], label='Acceleration X', alpha=0.8)
            plt.plot(data['Elapsed Time (ms)'], data['Acceleration Y'], label='Acceleration Y', alpha=0.8)
            plt.plot(data['Elapsed Time (ms)'], data['Acceleration Z'], label='Acceleration Z', alpha=0.8)
            
            # Add labels, legend, and grid
            plt.title(f"Acceleration Over Time - {file_name}")
            plt.xlabel("Elapsed Time (ms)")
            plt.ylabel("Acceleration (g)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Show the plot
            plt.show()
        else:
            print(f"File {file_name} does not contain the required columns.")
