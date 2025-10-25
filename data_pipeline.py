# data_pipeline.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler

def load_dataframe_from_upload(uploaded_file):
    """
    uploaded_file = streamlit uploaded file object or a file path string
    Returns: pandas.DataFrame
    """
    if isinstance(uploaded_file, str):
        df = pd.read_csv(uploaded_file)
    else:
        # streamlit file-like object
        df = pd.read_csv(uploaded_file)
    return df

def validate_target_column(df, possible_names=None):
    """
    Ensures a valid target column exists. Returns (target_name, df).
    Tries common names; raises ValueError with helpful message if none found.
    """
    possible = possible_names or ['Class', 'class', 'target', 'Label', 'label', 'fraud', 'is_fraud']
    for name in possible:
        if name in df.columns:
            return name, df
    raise ValueError(f"No target column found. Expected one of: {possible}. Columns present: {list(df.columns)}")

def encode_target(y):
    """
    Make sure y is numeric and binary (0/1).
    Accepts strings like 'Fraud'/'Non-Fraud' or 0/1, etc.
    Returns a cleaned pandas.Series of dtype int.
    """
    # If already numeric
    if pd.api.types.is_integer_dtype(y) or pd.api.types.is_float_dtype(y):
        y_clean = y.fillna(0).astype(int)
        unique_vals = np.unique(y_clean)
        if len(unique_vals) < 2:
            raise ValueError(f"Target has only one class after cleaning: {unique_vals}. Need at least 2 classes.")
        return y_clean

    # If object/string: try mapping common patterns
    y_str = y.astype(str).str.lower().str.strip()
    # common maps
    map_dict = {
        'fraud': 1, 'fraudulent': 1, '1': 1, 'true': 1, 'yes': 1,
        'non-fraud': 0, 'non fraud': 0, 'nonfraud': 0, 'not fraud': 0,
        'legit': 0, 'legitimate': 0, '0': 0, 'false': 0, 'no': 0
    }
    # try direct mapping
    mapped = y_str.map(map_dict)
    # if mapping produced NaNs, try fallback of labeling top value as 0 and others 1 if there are exactly two unique
    if mapped.isna().any():
        unique_vals = y_str.unique()
        if len(unique_vals) == 2:
            # assign 0 to most frequent, 1 to the other
            top = y_str.value_counts().idxmax()
            mapped = (y_str != top).astype(int)
        else:
            # try numeric conversion
            coerced = pd.to_numeric(y_str, errors='coerce')
            if coerced.notna().sum() / len(coerced) > 0.6:
                mapped = coerced.fillna(0).astype(int)
            else:
                # leave as-is: raise for clarity
                raise ValueError(f"Unable to automatically encode target values: {unique_vals}. Please convert to 0/1 or 'Fraud'/'Non-Fraud'.")
    mapped = mapped.fillna(0).astype(int)
    if mapped.nunique() < 2:
        raise ValueError(f"Target column encoded to a single class: {mapped.unique()}. Need at least 2 classes.")
    return mapped

def prepare_features(df, target_col, drop_columns=None):
    """
    Returns X (features DataFrame), y (Series)
    Drops non-numeric columns automatically. If you want to keep categorical features,
    you can one-hot encode them prior to calling this function.
    """
    df = df.copy()
    drop_columns = drop_columns or []
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from dataframe")
    y = df[target_col]
    X = df.drop(columns=[target_col] + drop_columns, errors='ignore')

    # Keep only numeric columns for baseline pipeline
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric feature columns found after dropping target. Provide numeric features or preprocess categoricals.")
    X_numeric = X[numeric_cols].fillna(0)
    return X_numeric, y

def normalize_features(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, scaler

def train_model(X, y, random_state=42):
    """
    Trains a RandomForestClassifier with basic params.
    Returns: model, metrics dict
    """
    # Ensure y is numeric and binary
    y_clean = encode_target(y)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_clean, test_size=0.2, random_state=random_state, stratify=y_clean)

    # normalize
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)

    # model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)

    # Try fit with friendly error handling
    try:
        model.fit(X_train_scaled, y_train)
    except Exception as e:
        raise ValueError(f"Failed to fit model: {e}")

    # predictions + metrics
    preds = model.predict(X_test_scaled)
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'classification_report': classification_report(y_test, preds, output_dict=True)
    }

    result = {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'feature_columns': X.columns.tolist()
    }
    return result

def run_pipeline_from_dataframe(df, target_names=None):
    """
    Top-level convenience function:
    - validate target
    - prepare features
    - train model and return result dict
    """
    target_col, df = validate_target_column(df, possible_names=target_names)
    X, y = prepare_features(df, target_col)
    result = train_model(X, y)
    # include some extra info
    result['target_col'] = target_col
    result['class_counts'] = df[target_col].value_counts().to_dict()
    return result
