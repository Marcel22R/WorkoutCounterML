import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_linAccData_processed_workoutCounter.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = df[df["set"] == 2]
plt.plot(set_df["acc_z"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_x"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = (
    df.query("label == 'Romanian-deadlift'").sort_values("participant").reset_index()
)
fix, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("participant")
ax.set_xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = "Romanian-deadlift"
participant = "Erwin"
all_axis_df = (
    df.query(f"label=='{label}'").query(f"participant=='{participant}'").reset_index()
)

fix, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label=='{label}'")
            .query(f"participant=='{participant}'")
            .reset_index()
        )
        if len(all_axis_df) > 0:
            fix, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("acc_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} {participant}".title())
            plt.legend()



# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = "Squats"
participant = "Marcel"
combined_plot_df = (
    df.query(f"label=='{label}'")
    .query(f"participant=='{participant}'")
    .reset_index(drop=True)
)
combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot()

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        combined_plot_df = (
            df.query(f"label=='{label}'")
            .query(f"participant=='{participant}'")
            .reset_index()
        )
        if len(combined_plot_df) > 0:
            plt.title(f"Exercise: {label.title()} Participant: {participant}")
            plt.legend()
            combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot()
            plt.savefig(f"../../reports/figures/{label.title()}_{participant}.png")
