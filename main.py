import pandas as pd
from argparse import ArgumentParser
from src import data_utils, visualization
from src.feature_engineering import preprocess_features, drop_features
from src import modeling

# Set up argument parser
parser = ArgumentParser(description="Train and evaluate models for defect prediction.")
parser.add_argument("-v","--verbose", action="store_true", help="Enable verbose output.")
args = parser.parse_args()

train_df, test_df = data_utils.load_data()

if args.verbose:
    print("Train and test data loaded successfully.")
    # Get Target details
    data_utils.target_variable_details(train_df)

    # Print feature summary table (optional)
    data_utils.feature_summmary_table(train_df)

# Visualize features
visualization.plot_features_histogram(train_df)

# Scale features
y_train = train_df["defects"]
train_df_clean = train_df.drop(columns=["defects", "id"])
X_train = preprocess_features(train_df_clean)

# Visualize scaled features before and after
visualization.compare_features_histogram(train_df_clean, X_train, y_train)

# Drop features
y_train_final = y_train.copy()
features_to_drop = ["locCodeAndComment", "lOBlank", "e", "t", "v", "b"]

X_train_final = drop_features(X_train, features_to_drop)

# Split data
X_train_split, X_val_split, X_test_split, y_train_split, y_val_split, y_test_split = (
    modeling.split_data(X_train_final, y_train_final, verbose=args.verbose)
)

# Train models
basic_lr_model = modeling.train_lr_model(X_train_split, y_train_split)
basic_rf_model = modeling.train_rf_basic(X_train_split, y_train_split)
best_rf_model = modeling.train_rf_tuned(X_train_split, y_train_split, verbose=True)

# Evaluate models
models = {
    "Logistic Regression": basic_lr_model,
    "Random Forest (Basic)": basic_rf_model,
    "Random Forest (Tuned)": best_rf_model,
}
if args.verbose:
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        print("-" * 30)
        print("Confusion Matrix and AUC curves saved to figs/model_evals/")
        print("-" * 30)
        print("Classification Report:")
        modeling.evaluate_model(model, X_val_split, y_val_split, model_name=model_name)
        print("\n")

# Compare models
if args.verbose:
    print("Comparing models...")
    print("-" * 30)
    modeling.compare_models(list(models.values()), X_val_split, y_val_split)

X_test_final = test_df.drop(columns=["id"])
submission_ids = test_df["id"]

X_test_final = preprocess_features(X_test_final)
X_test_final = drop_features(X_test_final, features_to_drop)

test_probabilities = best_rf_model.predict_proba(X_test_final)[:, 1]

submission_df = pd.DataFrame({
    "id": submission_ids,
    "defects": test_probabilities,
})
submission_df.to_csv("submission.csv", index=False)