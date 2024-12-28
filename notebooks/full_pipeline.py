import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from sklearn.neighbors import LocalOutlierFactor

from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix

import onnxruntime as rt


# Goes all the way from processed data to ort model

def main(relativeUrl, outlierMethod, myData):
    df = pd.read_pickle(relativeUrl)

    marcelLabels={"overhead":"Arnold-press", "bench":"Flat-bench-press", "back":"Lat-Pulldown", "pull":"Romanian-deadlift", "squat":"Squats", "break":"Break"}
    daveLabels={"overhead":"Ohp", "bench":"bench", "back":"Row", "pull":"Dead", "squat":"squat", "break":"Rest"}

    daveParticipants={"A":"A", "B":"B", "C":"C", "D":"D","E":"E"}
    marcelParticipants={"A": "Arzoo", "B":"Marcel", "C": "Pascal", "D": "Eugen","E": "Erwin"}

    daveTimeStamps={"timestamps": "epoch (ms)"}
    marcelTimeStamps={"timestamps": "Timestamp (ms)"}


    myLabels=daveLabels
    myParticipants=daveParticipants
    myTimeStamps=daveTimeStamps


    if(myData):
        myLabels=marcelLabels
        myParticipants=marcelParticipants
        myTimeStamps=marcelTimeStamps

    # --------------------------------------------------------------
    # Plot single columns
    # --------------------------------------------------------------

    set_df = df[df["set"] == 5]
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
        df.query(f"label == '{myLabels['bench']}'").sort_values("participant").reset_index()
    )
    fix, ax = plt.subplots()
    participant_df.groupby(["participant"])["acc_y"].plot()
    ax.set_ylabel("participant")
    ax.set_xlabel("samples")
    plt.legend()

    # --------------------------------------------------------------
    # Plot multiple axis
    # --------------------------------------------------------------

    label = myLabels['squat']
    participant = myParticipants['E']
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

    for label in labels:
        for participant in participants:
            all_axis_df = (
                df.query(f"label=='{label}'")
                .query(f"participant=='{participant}'")
                .reset_index()
            )
            if len(all_axis_df) > 0:
                fix, ax = plt.subplots()
                all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
                ax.set_ylabel("gyr_y")
                ax.set_xlabel("samples")
                plt.title(f"{label} {participant}".title())
                plt.legend()

    # --------------------------------------------------------------
    # Combine plots in one figure
    # --------------------------------------------------------------

    label = myLabels['squat']
    participant = myParticipants['B']
    combined_plot_df = (
        df.query(f"label=='{label}'")
        .query(f"participant=='{participant}'")
        .reset_index(drop=True)
    )
    fix, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
    combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
    combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

    ax[0].legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
    )
    ax[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
    )
    ax[1].set_xlabel("samples")
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
                fix, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
                combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
                combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

                ax[0].legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.15),
                    ncol=3,
                    fancybox=True,
                    shadow=True,
                )
                ax[1].legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.15),
                    ncol=3,
                    fancybox=True,
                    shadow=True,
                )
                ax[1].set_xlabel("samples")
                plt.suptitle(f"{label} {participant}".title(), y=0.95)
                plt.savefig(f"../reports/pipelineFigures/{label.title()}_{participant}.png")
                plt.show()

    df=pd.read_pickle(relativeUrl)

    outlier_columns=list(df.columns[:6])
    print(f"Outlier columns: {outlier_columns}")

    # --------------------------------------------------------------
    # Plotting outliers
    # --------------------------------------------------------------

    plt.style.use("fivethirtyeight")
    plt.rcParams["figure.figsize"] =(20,5)
    plt.rcParams["figure.dpi"]=100

    df[["acc_x", "label"]].boxplot(by="label",figsize=(20,10))
    df[outlier_columns[:3]+["label"]].boxplot(by="label",figsize=(20,10), layout=(1,3))
    df[outlier_columns[3:]+["label"]].boxplot(by="label",figsize=(20,10), layout=(1,3))

        # Plot a single column
    col = "acc_x"
    dataset= mark_outliers_iqr(df, col)
    plot_binary_outliers(dataset=dataset,col=col, outlier_col=col+"_outlier", reset_index=True)

    # Loop over all columns

    for col in outlier_columns:
        dataset= mark_outliers_iqr(df, col)
        plot_binary_outliers(dataset=dataset,col=col, outlier_col=col+"_outlier", reset_index=True)
    
    df[outlier_columns[:3]+["label"]].plot.hist(by="label",figsize=(20,10), layout=(3,3))
    df[outlier_columns[3:]+["label"]].plot.hist(by="label",figsize=(20,10), layout=(3,3))

    for col in outlier_columns:
        dataset= mark_outliers_chauvenet(df, col)
        plot_binary_outliers(dataset=dataset,col=col, outlier_col=col+"_outlier", reset_index=True)

    dataset, outliers, X_scores=mark_outliers_lof(df, outlier_columns)
    for col in outlier_columns:
        plot_binary_outliers(dataset=dataset,col=col, outlier_col="outlier_lof", reset_index=True)


    # --------------------------------------------------------------
    # Check outliers grouped by label
    # --------------------------------------------------------------
    label=myLabels['bench']

    for col in outlier_columns:
        dataset= mark_outliers_iqr(df[df["label"]==label], col)
        plot_binary_outliers(dataset=dataset,col=col, outlier_col=col+"_outlier", reset_index=True)

    label=myLabels['bench']

    for col in outlier_columns:
        dataset= mark_outliers_chauvenet(df[df["label"]==label], col)
        plot_binary_outliers(dataset=dataset,col=col, outlier_col=col+"_outlier", reset_index=True)

    dataset, outliers, X_scores=mark_outliers_lof(df[df["label"]==label], outlier_columns)
    for col in outlier_columns:
        plot_binary_outliers(dataset=dataset,col=col, outlier_col="outlier_lof", reset_index=True)

    col="gyr_z"
    dataset=mark_outliers_chauvenet(df, col=col)

    dataset[dataset["gyr_z_outlier"]]

    dataset.loc[dataset["gyr_z_outlier"], "gyr_z"]=np.nan

    # Create a loop
    outliers_removed_df=df.copy()
    for col in outlier_columns:
        for label in df["label"].unique():
            dataset= mark_outliers_iqr(df[df["label"]==label], col)
            dataset.loc[dataset[col+"_outlier"], col]=np.nan
            outliers_removed_df.loc[(outliers_removed_df["label"]==label),col]=dataset[col]
            n_outliers=len(dataset)-len(dataset[col].dropna())
            print(f"removed {n_outliers} outliers from {col} for {label}")

    outliers_removed_df

    outliers_removed_df.info()

    chosenMethod={"c":"chauvenet", "i": "iqr", "l": "lof"}
    predictor_columns = list(df.columns[:6])

    # --------------------------------------------------------------
    # Dealing with missing values (imputation)
    # --------------------------------------------------------------

    for col in predictor_columns:
        outliers_removed_df[col] = outliers_removed_df[col].interpolate()

    # --------------------------------------------------------------
    # Export new dataframe
    # --------------------------------------------------------------
    outliers_removed_df.to_pickle(f"../data/pipeline/02_outliers_removed_{chosenMethod[outlierMethod]}.pkl")

    df=outliers_removed_df


    plt.style.use("fivethirtyeight")
    plt.rcParams["figure.figsize"] = (20, 5)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["lines.linewidth"] = 2


    # --------------------------------------------------------------
    # Calculating set duration
    # --------------------------------------------------------------

    df[df["set"] == 28]["acc_y"].plot()


    duration = df[df["set"] == 2].index[-1] - df[df["set"] == 2].index[0]
    duration.seconds

    for s in df["set"].unique():
        start = df[df["set"] == s].index[0]
        stop = df[df["set"] == s].index[-1]

        duration = stop - start
        df.loc[(df["set"] == s), "duration"] = duration.seconds

    duration_df = df.groupby(["category"])["duration"].median()

    duration_df.iloc[0] / 5

    # --------------------------------------------------------------
    # Butterworth lowpass filter
    # --------------------------------------------------------------

    df_lowpass = df.copy()
    LowPass = LowPassFilter()


    fs = 1000 / 200
    cutoff = 1.3

    df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

    subset = df_lowpass[df_lowpass["set"] == 12]

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
    ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
    ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)


    for col in predictor_columns:
        df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
        df_lowpass[col] = df_lowpass[col + "_lowpass"]
        del df_lowpass[col + "_lowpass"]


    # --------------------------------------------------------------
    # Principal component analysis PCA
    # --------------------------------------------------------------

    df_pca = df_lowpass.copy()
    PCA = PrincipalComponentAnalysis()

    pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

    plt.figure(figsize=(10, 10))
    plt.plot(range(1, len(predictor_columns) + 1), pc_values)
    plt.xlabel("principal component number")
    plt.ylabel("explained variance")
    plt.show()

    df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

    subset = df_pca[df_pca["set"] == 12]

    subset[["pca_1", "pca_2", "pca_3"]].plot()

    # --------------------------------------------------------------
    # Sum of squares attributes
    # --------------------------------------------------------------

    df_squared = df_pca.copy()

    acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
    gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

    df_squared["acc_r"] = np.sqrt(acc_r)
    df_squared["gyr_r"] = np.sqrt(gyr_r)

    subset = df_squared[df_squared["set"] == 12]
    subset[["acc_r", "gyr_r"]].plot(subplots=True)


    # --------------------------------------------------------------
    # Temporal abstraction
    # --------------------------------------------------------------

    df_temporal = df_squared.copy()
    NumAbs = NumericalAbstraction()

    predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

    ws = int(1000 / 200)

    for col in predictor_columns:
        df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
        df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")

    df_temporal_list = []

    for s in df_temporal["set"].unique():
        subset = df_temporal[df_temporal["set"] == s].copy()
        for col in predictor_columns:
            subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
            subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
        df_temporal_list.append(subset)

    df_temporal = pd.concat(df_temporal_list)

    subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
    subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()


    # --------------------------------------------------------------
    # Frequency features
    # --------------------------------------------------------------

    df_freq = df_temporal.copy().reset_index()
    FreqAbs = FourierTransformation()


    fs = int(1000 / 200)
    ws = int(2800 / 200)

    df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)
    df_freq.info()

    df_freq.columns
    subset = df_freq[df_freq["set"] == 30]
    subset[["acc_y"]].plot()
    subset[
        [
            "acc_y_max_freq",
            "acc_y_freq_weighted",
            "acc_y_pse",
            "acc_y_freq_1.429_Hz_ws_14",
            "acc_y_freq_2.5_Hz_ws_14",
        ]
    ].plot()

    df_freq_list = []
    for s in df_freq["set"].unique():
        print(f"Applying fourier transformations to set {s}")
        subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
        subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
        df_freq_list.append(subset)


    

    df_freq = pd.concat(df_freq_list).set_index(myTimeStamps['timestamps'], drop=True)


    # --------------------------------------------------------------
    # Dealing with overlapping windows
    # --------------------------------------------------------------

    df_freq = df_freq.dropna()
    df_freq.iloc[::2]

    # --------------------------------------------------------------
    # Clustering
    # --------------------------------------------------------------

    df_cluster = df_freq.copy()

    cluster_columns = ["acc_x", "acc_y", "acc_z"]
    k_values = range(2, 10)
    inertias = []

    for k in k_values:
        subset = df_cluster[cluster_columns]
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
        cluster_labels = kmeans.fit_predict(subset)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(10,10))
    plt.plot(k_values, inertias)
    plt.xlabel("k")
    plt.ylabel("Sum of squared distances")
    plt.show()


    kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
    subset = df_cluster[cluster_columns]
    df_cluster["cluster"] = kmeans.fit_predict(subset)


    fig=plt.figure(figsize=(15,15))
    ax= fig.add_subplot(projection="3d")
    for c in df_cluster["cluster"].unique():
        subset=df_cluster[df_cluster["cluster"]==c]
        ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.legend()
    plt.show()


    fig=plt.figure(figsize=(15,15))
    ax= fig.add_subplot(projection="3d")
    for c in df_cluster["label"].unique():
        subset=df_cluster[df_cluster["label"]==c]
        ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.legend()
    plt.show()

    # --------------------------------------------------------------
    # Export dataset
    # --------------------------------------------------------------

    df_cluster.to_pickle("../data/pipeline/03_data_features.pkl")
    df_cluster.to_csv("../data/pipeline/03_data_features.csv")

    # Plot settings
    plt.style.use("fivethirtyeight")
    plt.rcParams["figure.figsize"] = (20, 5)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["lines.linewidth"] = 2

    df = pd.read_pickle("../data/pipeline/03_data_features.pkl")


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


    fig, ax = plt.subplots(figsize=(10, 5))
    df_train["label"].value_counts().plot(
        kind="bar", ax=ax, color="lightblue", label="Total"
    )
    Y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
    Y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
    plt.legend()
    plt.show()


    # --------------------------------------------------------------
    # Split feature subsets
    # --------------------------------------------------------------

    basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    square_features = ["acc_r", "gyr_r"]
    pca_features = ["pca_1", "pca_2", "pca_3"]
    time_features = [f for f in df_train.columns if "_temp_" in f]
    freq_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]
    cluster_features = ["cluster"]

    print("Basic features:", len(basic_features))
    print("Square features:", len(square_features))
    print("Pca features:", len(pca_features))
    print("Time features:", len(time_features))
    print("frequency features:", len(freq_features))
    print("Cluster features:", len(cluster_features))

    feature_set_1 = list(set(basic_features))
    feature_set_2 = list(set(basic_features + square_features + pca_features))
    feature_set_3 = list(set(feature_set_2 + time_features))
    feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))


    # --------------------------------------------------------------
    # Perform forward feature selection using simple decision tree
    # --------------------------------------------------------------

    learner = ClassificationAlgorithms()

    max_features = 10
    selected_features, ordered_features, ordered_scores = learner.forward_selection(
         max_features, X_train, Y_train
     )
    # selected_features = [
    #     "duration",
    #     "acc_x",
    #     "pca_2",
    #     "gyr_r_freq_0.0_Hz_ws_14",
    #     "gyr_r_temp_std_ws_5",
    #     "gyr_y",
    #     "gyr_z_freq_2.143_Hz_ws_14",
    #     "acc_y_freq_0.714_Hz_ws_14",
    #     "gyr_r_freq_1.786_Hz_ws_14",
    #     "gyr_r_freq_1.071_Hz_ws_14",
    # ]


    # ordered_scores = [
    #     0.787715649088462,
    #     0.9635384803621682,
    #     0.9904563807659367,
    #     0.9963293772176679,
    #     0.9974305640523675,
    #     0.9977976263306008,
    #     0.9980423345160896,
    #     0.9980423345160896,
    #     0.9980423345160896,
    #     0.9980423345160896,
    # ]


    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, max_features + 1, 1), ordered_scores)
    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(1, max_features + 1, 1))
    plt.show()


    # --------------------------------------------------------------
    # Grid search for best hyperparameters and model selection
    # --------------------------------------------------------------

    possible_feature_sets = [
        feature_set_1,
        feature_set_2,
        feature_set_3,
        feature_set_4,
        selected_features,
    ]

    feature_names = [
        "Feature Set 1",
        "Feature Set 2",
        "Feature Set 3",
        "Feature Set 4",
        "Selected Features",
    ]

    iterations = 1
    score_df = pd.DataFrame()

    df_train.info(verbose=True)


    for i, f in zip(range(len(possible_feature_sets)), feature_names):
        print("Feature set:", i)
        selected_train_X = X_train[possible_feature_sets[i]]
        selected_test_X = X_test[possible_feature_sets[i]]

        # First run non deterministic classifiers to average their score.
        performance_test_nn = 0
        performance_test_rf = 0

        for it in range(0, iterations):
            print("\tTraining neural network,", it)
            (
                class_train_y,
                class_test_y,
                class_train_prob_y,
                class_test_prob_y,
            ) = learner.feedforward_neural_network(
                selected_train_X,
                Y_train,
                selected_test_X,
                gridsearch=False,
                saveModel=True
            )
            performance_test_nn += accuracy_score(Y_test, class_test_y)

            print("\tTraining random forest,", it)
            (
                class_train_y,
                class_test_y,
                class_train_prob_y,
                class_test_prob_y,
            ) = learner.random_forest(
                selected_train_X, Y_train, selected_test_X, gridsearch=True, saveModel=True
            )
            performance_test_rf += accuracy_score(Y_test, class_test_y)

        performance_test_nn = performance_test_nn / iterations
        performance_test_rf = performance_test_rf / iterations

        # And we run our deterministic classifiers:
        print("\tTraining KNN")
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.k_nearest_neighbor(
            selected_train_X, Y_train, selected_test_X, gridsearch=True
        )
        performance_test_knn = accuracy_score(Y_test, class_test_y)

        print("\tTraining decision tree")
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.decision_tree(
            selected_train_X, Y_train, selected_test_X, gridsearch=True,saveModel=True
        )
        performance_test_dt = accuracy_score(Y_test, class_test_y)

        print("\tTraining naive bayes")
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.naive_bayes(selected_train_X, Y_train, selected_test_X)

        performance_test_nb = accuracy_score(Y_test, class_test_y)

        print("\tTraining support vector machine")
       
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.support_vector_machine_without_kernel(selected_train_X, Y_train, selected_test_X)

        performance_test_svm=accuracy_score(Y_test, class_test_y)

        # Save results to dataframe
        models = ["NN", "RF", "KNN", "DT", "NB", "SVM"]
        new_scores = pd.DataFrame(
            {
                "model": models,
                "feature_set": f,
                "accuracy": [
                    performance_test_nn,
                    performance_test_rf,
                    performance_test_knn,
                    performance_test_dt,
                    performance_test_nb,
                    performance_test_svm,
                ],
            }
        )
        score_df = pd.concat([score_df, new_scores])


    # --------------------------------------------------------------
    # Create a grouped bar plot to compare the results
    # --------------------------------------------------------------

    score_df.sort_values(by="accuracy", ascending=False)

    plt.figure(figsize=(10, 10))
    sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0.7, 1)
    plt.legend(loc="lower right")
    plt.show()


    # --------------------------------------------------------------
    # Select best model and evaluate results
    # --------------------------------------------------------------


    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.random_forest(
        X_train[feature_set_4], Y_train, X_test[feature_set_4], gridsearch=True
    )

    accuracy=accuracy_score(Y_test, class_test_y)

    classes=class_test_prob_y.columns
    cm=confusion_matrix(Y_test, class_test_y, labels=classes)


    # create confusion matrix for cm
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.grid(False)
    plt.show()


    # --------------------------------------------------------------
    # Select train and test data based on participant
    # --------------------------------------------------------------

    participant_df=df.drop(["set", "category"],axis=1)
    X_train=participant_df[participant_df["participant"]!=myParticipants['D']].drop("label",axis=1)
    Y_train=participant_df[participant_df["participant"]!=myParticipants['D']]["label"]

    X_test=participant_df[participant_df["participant"]==myParticipants['D']].drop("label",axis=1)
    Y_test=participant_df[participant_df["participant"]==myParticipants['D']]["label"]

    X_train=X_train.drop(["participant"],axis=1)
    X_test=X_test.drop(["participant"],axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    df_train["label"].value_counts().plot(
        kind="bar", ax=ax, color="lightblue", label="Total"
    )
    Y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
    Y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
    plt.legend()
    plt.show()




    # --------------------------------------------------------------
    # Use best model again and evaluate results
    # --------------------------------------------------------------

    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.random_forest(
        X_train[feature_set_4], Y_train, X_test[feature_set_4], gridsearch=True
    )

    accuracy=accuracy_score(Y_test, class_test_y)

    classes=class_test_prob_y.columns
    cm=confusion_matrix(Y_test, class_test_y, labels=classes)


    # create confusion matrix for cm
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix Random forest. Training Set 4 of 5 people")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.grid(False)
    plt.show()


    # --------------------------------------------------------------
    # Try a simpler model with the selected features
    # --------------------------------------------------------------

    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.feedforward_neural_network(
        X_train[feature_set_4], Y_train, X_test[feature_set_4], gridsearch=True
    )

    accuracy=accuracy_score(Y_test, class_test_y)

    classes=class_test_prob_y.columns
    cm=confusion_matrix(Y_test, class_test_y, labels=classes)


    # create confusion matrix for cm
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix neural network")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.grid(False)
    plt.show()


 









def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()

def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset

def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset

def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.
    
    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores




relativeUrlWorkoutCounter="../data/interim/01_data_processed_workoutCounter.pkl"
relativeUrlDave="../data/interim/01_data_processed.pkl"
outlierMethod="c"
main(relativeUrlWorkoutCounter, outlierMethod, myData=True)
