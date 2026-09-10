
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    cols_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df_clean = df.drop(columns=cols_to_drop)
    df_clean.columns = (
        df_clean.columns
        .str.replace('[', '_', regex=False)
        .str.replace(']', '_', regex=False)
        .str.replace('<', '_', regex=False)
    )
    return df_clean


def prepare_train_test(df_clean):
    df_encoded = pd.get_dummies(df_clean, columns=['Type'], drop_first=True)

    X = df_encoded.drop(columns=['Machine failure'])
    y = df_encoded['Machine failure']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_cols = [
        'Air temperature _K_', 'Process temperature _K_',
        'Rotational speed _rpm_', 'Torque _Nm_', 'Tool wear _min_'
    ]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    return X_train_resampled, y_train_resampled, X_test_scaled, y_test, scaler


def train_final_model(X_train_resampled, y_train_resampled):
    best_params = {
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.2,
        'random_state': 42
    }
    model = XGBClassifier(**best_params)
    model.fit(X_train_resampled, y_train_resampled)
    return model


def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")


if __name__ == "__main__":
    DATA_PATH = "../data/ai4i2020.csv"
    MODEL_OUTPUT_PATH = "../models/xgb_final_model.joblib"
    SCALER_OUTPUT_PATH = "../models/scaler.joblib"

    print("مرحله ۱: بارگذاری و پاکسازی داده...")
    df_clean = load_and_clean_data(DATA_PATH)

    print("مرحله ۲: آماده‌سازی Train/Test...")
    X_train_res, y_train_res, X_test_scaled, y_test, scaler = prepare_train_test(df_clean)

    print("مرحله ۳: آموزش مدل نهایی...")
    final_model = train_final_model(X_train_res, y_train_res)

    print("مرحله ۴: ارزیابی نهایی...")
    evaluate_model(final_model, X_test_scaled, y_test)

    print("مرحله ۵: ذخیره مدل و اسکیلر...")
    joblib.dump(final_model, MODEL_OUTPUT_PATH)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)
    print(f"مدل ذخیره شد در: {MODEL_OUTPUT_PATH}")