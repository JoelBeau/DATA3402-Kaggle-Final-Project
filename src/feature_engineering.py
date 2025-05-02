import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_features(df):
    X_log_scaled = df.copy()
    for col in df.columns:
        X_log_scaled[col] = np.log1p(df[col])
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_log_scaled), columns=df.columns)
    return X_scaled

def drop_features(X, features_to_drop):
    return X.drop(columns=features_to_drop)
