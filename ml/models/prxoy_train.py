import pandas as pd
import os
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

from lightgbm import LGBMClassifier
from imblearn.over_sampling import BorderlineSMOTE, SMOTE, ADASYN

# ========== CONFIG ==========
TARGET_COL = 'is_proxy'
MODEL_NAME = 'model_lgbm.joblib'
ENCODERS_NAME = 'label_encoders_lgbm.joblib'
FEATURES_NAME = 'features.joblib'
# ============================


# ========= LOAD DATA =========
def load_data(csv_path):
    logging.info("Loading CSV data...")
    return pd.read_csv(csv_path, keep_default_na=False)


# ========= PREPROCESS =========
def preprocess_data(df, target_col):
    logging.info("Encoding categorical features...")
    label_encoders = {}

    for col in df.select_dtypes(include=['object']).columns:
        if col == target_col:
            continue

        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders


# ========= HANDLE IMBALANCE =========
def handle_class_imbalance(X, y, technique=None):
    if technique is None:
        logging.info("Skipping class imbalance...")
        return X, y

    logging.info(f"Applying {technique}...")

    if technique == 'ADASYN':
        sampler = ADASYN(random_state=42)
    elif technique == 'SMOTE':
        sampler = SMOTE(random_state=42)
    elif technique == 'BSMOTE':
        sampler = BorderlineSMOTE(random_state=42)
    else:
        raise ValueError("Invalid technique")

    X_res, y_res = sampler.fit_resample(X, y)
    logging.info(f"Balanced: {Counter(y_res)}")
    return X_res, y_res


# ========= MODEL =========
def train_model(X_train, y_train):
    logging.info("Training LightGBM model...")

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=10,
        num_leaves=31,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


# ========= SAVE =========
def save_artifacts(model, encoders, features, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model, os.path.join(output_dir, MODEL_NAME))
    joblib.dump(encoders, os.path.join(output_dir, ENCODERS_NAME))
    joblib.dump(features, os.path.join(output_dir, FEATURES_NAME))

    logging.info(f"Artifacts saved in {output_dir}")


# ========= PIPELINE =========
def train_pipeline(data_path, target_col, test_size=0.2,
                   balance_technique=None, output_dir='./'):

    df = load_data(data_path)
    df, label_encoders = preprocess_data(df, target_col)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Save feature columns
    feature_columns = X.columns.tolist()

    X_res, y_res = handle_class_imbalance(X, y, balance_technique)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=test_size,
        random_state=42, stratify=y_res
    )

    model = train_model(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    print("\n===== MODEL EVALUATION =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    save_artifacts(model, label_encoders, feature_columns, output_dir)

    return model


# ========= MAIN =========
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, r"C:\Users\kk919\Desktop\proxy_detection_model\ml\data\proxy_data.csv")

    train_pipeline(
        data_path=dataset_path,
        target_col=TARGET_COL,
        balance_technique='BSMOTE',
        output_dir=current_dir
    )