![](UTA-DataScience-Logo.png)

# Software Defect Prediction (Kaggle Playground S3E23)

**This repository contains a solution for the Kaggle challenge: [Playground Series - Season 3, Episode 23](https://www.kaggle.com/competitions/playground-series-s3e23), which involves predicting software defects using code metric features.**

---

## Overview

This Kaggle challenge is to build a binary classification model to predict whether a software file contains a defect (`1`) or not (`0`) using a range of numerical features extracted from code complexity and structure.

My approach involved exploring and comparing several machine learning models, including:
- Logistic Regression (baseline)
- Random Forest (primary model)
- Class balancing
- Feature transformation (log-scaling + standardization)

My best model (according to Kaggle's scoring metric of AUC)  —  a basic tuned Random Forest Classifier using RandomSearchCV, achieved a **validation accuracy of 81%**, with the best **AUC of 0.7885**. These results represent a solid baseline and demonstrate the importance of preprocessing and class imbalance strategies.

---

## Summary of Work Done

### Data

- **Type**: Tabular data in `.csv` format.
  - Input: Numeric software metrics (e.g., lines of code, branching, operands)
  - Output: Binary target `defects` column
- **Size**:
  - Training set: 100,000+ rows
  - Test set: 68,000+ rows
- **Split**:
  - 60% Training
  - 20% Validation
  - 20% Test (local)

#### Preprocessing / Clean up

- Dropped clearly uninformative or noisy features (`locCodeAndComment`, `IOBlank`, `id`, etc.)
- Applied `np.log1p()` to reduce skewness
- Standardized features using `StandardScaler`
- Handled class imbalance using `class_weight='balanced'`

#### Data Visualization

To better understand the underlying structure and separability of the dataset, several visualization techniques were applied:

#### Class-Wise Feature Distributions

- For each feature, I plotted histograms comparing the distribution of values for each class (`defects = 0` vs. `defects = 1`).
- These visualizations revealed that many features were **right-skewed**, and raw values often had poor visual separation between classes.
- Features like `l`, `branchCount`, `v(g)`, and `total_Opnd` showed **visually distinct distributions** between the classes, especially after transformation, suggesting they may be useful predictors.

#### Distribution Comparison (Before vs. After Scaling)

- Applied **log1p transformation** followed by **standard scaling** to every feature and visualized both the raw and scaled versions side-by-side (example below).
  ![image](https://github.com/user-attachments/assets/e8470493-62ce-4470-b675-dd2be21ab698)
- This allowed us to see:
  - Which features had long-tailed distributions that could obscure class separation.
  - Whether scaling improved class separability (for example, `uniq_Op`, `total_Op`, and `IOCode` showed clearer differences post-scaling).
- Even features that initially looked noisy or overlapping (like `IOBlank`, `e`, `t`) were retained temporarily for inspection after transformation, though many of these were ultimately dropped based on post-scaling insights.

#### Why I Chose to Scale

- While tree-based models like Random Forest don’t *require* feature scaling, we scaled anyway to:
  - Better **visualize class separation**
  - Maintain **consistency** across models (e.g., Logistic Regression, which is sensitive to scaling)
- Scaling helped **uncover hidden structure** in features that initially appeared flat or noisy, such as `iv(g)` and `IOComment`.

---

### Problem Formulation

- **Input**: 1D vector of numeric code metrics
- **Output**: Binary label — `defects`: 1 = defective, 0 = clean
- **Model**: RandomForestClassifier (`sklearn`) with calibrated probabilities
- **Loss Function**: `log_loss` (Kaggle metric)
- **Hyperparameters**:
  - `n_estimators = 200`
  - `max_depth = None`
  - `class_weight = 'balanced'`

---

### Training

- Trained using `scikit-learn` on a standard CPU machine (8-core, 16 GB RAM)
- Training time: ~2 minute
- Early stopping not applicable; Random Forest is non-iterative
- Trained and validated on fixed splits with `random_state=42`

---

### Performance Comparison

| Model                                                | Accuracy | Log Loss | AUC  |
|------------------------------------------------------|----------|----------|------|
| Logistic Regression                                  | 75%      | 0.55     | 0.78 |
| Random Forest (basic)                                | 81%      | 0.46     | 0.77 |
| Random Forest (tuned attempt with RandomizedSearchCV)| 81%      | 0.52     | 0.79 |

---

### Conclusions

- Random Forests performed best with minimal tuning
- Preprocessing (log-scaling and feature selection) made a large impact:
- Threshold tuning significantly improved defect recall
- Class weighting helped balance metrics on imbalanced data

---

### Future Work

- Try XGBoost or LightGBM with tuned parameters
- Use SHAP values for better interpretability
- Test the ensemble of models to improve log loss
- Explore feature engineering based on domain knowledge

---

##  Overview of Files in Repository

### Directory Structure
  ```
  .
  ├── data/ # Raw input files (train.csv, test.csv)
  ├── models/ # Saved model files (.pkl.gz) for reuse or submission
  ├── notebooks/ # Jupyter notebooks used for initial development & notes
  │ └── Kaggle Tabular Data.ipynb
  ├── src/ # Core Python modules
  │   ├── data_utils.py
  │   ├── feature_engineering.py
  │   ├── modeling.py
  │   ├── thresholds.py
  │   └── visualization.py
  ├── main.py # Main script to run the entire training + prediction pipeline
  ├── requirements.txt # List of required Python packages
  └── README.md # Project description and instructions
  ```

### File Descriptions
- `data/`: Contains the raw input files (`train.csv`, `test.csv`) downloaded from Kaggle.
- `notebooks/Kaggle Tabular Data.ipynb`: Exploratory notebook used during development and visualization. Also contains my thoughts and notes about why I made certain decisions e.g., why I chose to scale, why I dropped certain features, etc.
- `src/figures/`: Contains the figures generated during the analysis and visualization steps.
- `src/data_utils.py`: Loads and inspects raw `.csv` data files.
- `src/feature_engineering.py`: Performs log scaling, feature dropping, and standardization.
- `src/modeling.py`: Contains model training and evaluation logic (Random Forest, Logistic Regression).
- `src/visualization.py`: Creates distribution plots, histograms, and feature comparisons by defect class. Both raw and scaled versions are included.
- `src/models/`: Folder for saved model binaries (e.g., `.pkl.gz`) used for evaluation or submission.
- `main.py`: Main driver script that ties together all preprocessing, training, evaluation, and submission steps.
- `requirements.txt`: Python dependencies for the project.
- `submission.csv`: Example submission file for Kaggle competition.
- `README.md`: This file, providing an overview of the project and instructions for use.

---

## How to Reproduce Results

1. Clone this repository
2. Install Python dependencies from `requirements.txt`
3. Download `train.csv` and `test.csv` from [Kaggle S3E23](https://www.kaggle.com/competitions/playground-series-s3e23)
4. Run `main.py` or execute steps inside `Kaggle Tabular Data.ipynb`
5. Submit `submission.csv` to the Kaggle competition

