import matplotlib.pyplot as plt
import numpy as np
import os
import math


def plot_features_histogram(train_df):
    # Get numerical features only (excluding target)
    numerical_features = (
        train_df.select_dtypes(include=["int64", "float64", "bool"])
        .drop(columns=["defects", "id"])
        .columns
    )

    # Grid config
    n_cols = 3
    n_rows = math.ceil(len(numerical_features) / n_cols)

    # Plot setup
    plt.figure(figsize=(n_cols * 5, n_rows * 3))

    for idx, feature in enumerate(numerical_features, 1):
        plt.subplot(n_rows, n_cols, idx)
        # Plot histogram for non-defective class (defects = 0)
        plt.hist(
            train_df[train_df["defects"] == 0][feature],
            bins=30,
            alpha=0.5,
            label="No Defect",
            color="blue",
            density=True,
        )

        # Plot histogram for defective class (defects = 1)
        plt.hist(
            train_df[train_df["defects"] == 1][feature],
            bins=30,
            alpha=0.5,
            label="Defect",
            color="red",
            density=True,
        )

        # Add title and labels
        plt.title(feature)
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend(loc="upper right")
        plt.tight_layout()

    plt.suptitle(
        "Distribution of Numerical Features by Defect Class", fontsize=16, y=1.02
    )

    # Get the absolute path to the 'figs' folder
    figs_dir = os.path.join(os.path.dirname(__file__), "figs")

    # Save the figure
    plt.savefig(os.path.join(figs_dir, "all_feature_histograms.png"))


def compare_features_histogram(train_df_clean, X_train, y_train):
    # Add target back to original and scaled feature sets
    X_with_target = train_df_clean.copy()
    X_with_target["defects"] = y_train

    X_scaled_with_target = X_train.copy()
    X_scaled_with_target["defects"] = y_train

    # Config
    features_per_fig = 4
    features = X_train.columns.tolist()

    for i in range(0, len(features), features_per_fig):
        feature_group = features[i : i + features_per_fig]
        n_rows = len(feature_group)

        fig, axs = plt.subplots(n_rows, 2, figsize=(12, 3 * n_rows))

        # If only one feature, axs will be 1D — force it into 2D
        if n_rows == 1:
            axs = np.expand_dims(axs, axis=0)

        for j, feature in enumerate(feature_group):
            # Original
            axs[j][0].hist(
                [
                    X_with_target[X_with_target["defects"] == 0][feature],
                    X_with_target[X_with_target["defects"] == 1][feature],
                ],
                bins=30,
                alpha=0.5,
                label=["No Defect", "Defect"],
                color=["blue", "red"],
                density=True,
            )
            axs[j][0].set_title(f"Original: {feature}")
            axs[j][0].set_xlabel("")
            axs[j][0].legend()

            # Scaled
            axs[j][1].hist(
                [
                    X_scaled_with_target[X_scaled_with_target["defects"] == 0][feature],
                    X_scaled_with_target[X_scaled_with_target["defects"] == 1][feature],
                ],
                bins=30,
                alpha=0.5,
                label=["No Defect", "Defect"],
                color=["blue", "red"],
                density=True,
            )
            axs[j][1].set_title(f"Scaled: {feature}")
            axs[j][1].set_xlabel("")
            axs[j][1].legend()

        plt.tight_layout()

        figs_dir = os.path.join(os.path.dirname(__file__), "figs")
        plt.savefig(
            os.path.join(figs_dir, f"feature_histograms_{i//features_per_fig}.png")
        )
