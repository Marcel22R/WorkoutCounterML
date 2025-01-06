import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
accuracy_score = pd.DataFrame({
    "paper": ["Paper 1", "Paper 2", "Paper 3", "This paper"],
    "accuracy": [0.98, 0.82, 0.93, 0.97]
})

# Plot
plt.figure(figsize=(8, 6))  # Adjust figure size for better proportions
sns.barplot(
    x="paper", 
    y="accuracy", 
    data=accuracy_score, 
    palette="Blues_d",  # Add a visually appealing color palette
    edgecolor="black"  # Add borders to bars
)

# Customizations

plt.ylabel("Accuracy", fontsize=12)
plt.xlabel("", fontsize=12)
plt.ylim(0.5, 1)
plt.grid(axis="y", linestyle="--", alpha=0.7)  # Add gridlines for better readability

# Remove legend (optional here since it's not needed)
# If needed in other plots, `plt.legend()` can be used

# Adjust tick label size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Display the plot
plt.tight_layout()  # Ensure everything fits within the figure
plt.show()
