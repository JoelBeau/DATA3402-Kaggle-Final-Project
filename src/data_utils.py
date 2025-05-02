import pandas as pd
import tabulate


def load_data(train_path="data/train.csv", test_path="data/test.csv"):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def check_missing(df):
    return df.isnull().sum()


def feature_summmary_table(df):

    # Exclude the target column from features
    features = df.drop(columns=["defects"])

    # Initialize the summary table
    summary_table = []

    for col in features.columns:
        col_data = df[col]
        # Determine the type of data (categorical or numerical)
        # and calculate statistics accordingly
        data_type = "Categorical" if col_data.nunique() < 10 else "Numerical"
        unique_values = (
            col_data.unique()
            if data_type == "Categorical"
            else f"{col_data.min()} to {col_data.max()}"
        )
        missing = col_data.isnull().sum()
        mean = col_data.mean()
        std = col_data.std()

        # Calculate outliers using the 3-sigma rule
        # (values outside mean ± 3 * std are considered outliers)
        outliers = ((col_data < (mean - 3 * std)) | (col_data > (mean + 3 * std))).sum()

        summary_table.append(
            {
                "Feature": col,
                "Type": data_type,
                "Values": (
                    unique_values
                    if data_type == "Categorical"
                    else f"{col_data.min()} to {col_data.max()}"
                ),
                "Missing Values": missing,
                "Summation of Outliers": outliers,
            }
        )

    # Convert the summary table to a DataFrame for better visualization
    summary_df = pd.DataFrame(summary_table)

    # Display the summary table
    print("Summary of features:")
    print(
        tabulate.tabulate(summary_df, headers="keys", tablefmt="psql", showindex=False)
    )

def target_variable_details(df):
    target_counts = df['defects'].value_counts()
    print("Target variable counts:")
    print(target_counts)

    # Display the range of the target variable
    target_range = (df['defects'].min(), df['defects'].max())
    print(f"Target variable range: {target_range}")
