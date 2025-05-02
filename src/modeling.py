import numpy as np
import os
import joblib
from matplotlib import pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss
from tabulate import tabulate

# Directory for saving models
models_dir = os.path.join(os.path.dirname(__file__), "models")

def split_data(X, target, verbose=False):
    # First split: 60% train, 40% temp (to later split into val and test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, target, test_size=0.4, random_state=42, stratify=target
    )

    # Second split: 50% of temp → val and test (i.e. 20% each of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    if verbose:
        # Check the shapes of the splits
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_lr_model(X_train, y_train):

    # Logistic Regression basic model
    basic_lr_model = LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )
    basic_lr_model.fit(X_train, y_train)

    # Save the model
    joblib.dump(basic_lr_model, os.path.join(models_dir,"basic_lr_model.pkl"))

    return basic_lr_model


def train_rf_basic(X_train, y_train):
    # Initialize and train with class_weight to handle imbalance
    basic_rf_model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    basic_rf_model.fit(X_train, y_train)

    # Save the model
    joblib.dump(basic_rf_model, os.path.join(models_dir,"basic_rf_model.pkl"))

    return basic_rf_model


def train_rf_tuned(X_train, y_train, verbose=False):
    # Define the parameter distribution
    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced"],
    }

    # Initialize the model
    rf = RandomForestClassifier(random_state=42)

    # Perform randomized search
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=10,  # Number of parameter settings sampled
        cv=5,  # 5-fold cross-validation
        scoring="roc_auc",  # Use log loss as the scoring metric
        n_jobs=-1,
        random_state=42,
        verbose=2 if verbose else 0,
    )

    # Fit the randomized search to the data
    random_search.fit(X_train, y_train)

    # Use the best model
    best_rf_model = random_search.best_estimator_

    if verbose:
        print(f"Best parameters for random forest model: {random_search.best_params_}")

    # Save the best model
    joblib.dump(best_rf_model, os.path.join(models_dir,"best_rf_model.pkl"))

    return best_rf_model


def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.colorbar()
    tick_marks = range(len(set(y_true)))
    plt.xticks(tick_marks, set(y_true), rotation=45)
    plt.yticks(tick_marks, set(y_true))

    # Normalize the confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot normalized values
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(
                j,
                i,
                f"{cm_normalized[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_normalized[i, j] > 0.5 else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    
    model_evals_dir = os.path.join(os.path.dirname(__file__), "figs/model_evals")

    plt.savefig(os.path.join(model_evals_dir, f"{model_name}_confusion_matrix.png"))

def evaluate_model(model, X, y, model_name="Model"):
    # Predict on the validation set
    y_pred = model.predict(X)

    print(f"{model_name} evaluation metrics:")

    # Print classification report
    print(classification_report(y, y_pred))

    # Plot confusion matrix
    plot_confusion_matrix(y, y_pred, model_name=model_name)

    # Plot ROC curve
    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend(loc="lower right")

    model_evals_dir = os.path.join(os.path.dirname(__file__), "figs/model_evals")

    plt.savefig(os.path.join(model_evals_dir, f"{model_name}_roc_curve.png"))


def model_auc(model, X, y):
    # Predict probabilities
    y_proba = model.predict_proba(X)[:, 1]  # Get probabilities for the positive class

    # Calculate AUC
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def model_log_loss(model, X, y):
    # Predict probabilities
    y_proba = model.predict_proba(X)[:, 1]  # Get probabilities for the positive class

    # Calculate log loss
    log_loss_value = log_loss(y, y_proba)
    return log_loss_value


def compare_models(models, X_val, y_val):
    names = ["Logistic Regression", "Basic Random Forest", "Tuned, Random Forest"]

    # Example data for tabulation
    results = [["Model", "ROC AUC", "Log Loss"]]

    # Create Table
    for m, n in zip(models, names):
        ll = f"{model_log_loss(m, X_val, y_val):.4f}"
        area = f"{model_auc(m, X_val, y_val):.4f}"
        results.append([n, area, ll])

    # Print the table
    print(tabulate(results, headers="firstrow", tablefmt="grid"))
