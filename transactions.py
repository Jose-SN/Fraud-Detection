import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import warnings
from tqdm import tqdm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


warnings.filterwarnings("ignore")

# Load dataset
data_dir = "./assets"
train_transaction_path = os.path.join(data_dir, "train_transaction.csv")
train_identity_path = os.path.join(data_dir, "train_identity.csv")

if not os.path.exists(train_transaction_path) or not os.path.exists(train_identity_path):
    raise FileNotFoundError("One or both dataset files are missing. Please check the file paths.")

df_train = pd.read_csv(train_transaction_path)
df_identity = pd.read_csv(train_identity_path)

# Merge datasets
df = df_train.merge(df_identity, on="TransactionID", how="left")

# Handle missing values
df.fillna(-999, inplace=True)

# Encode categorical variables
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Feature selection
features = [col for col in df.columns if col not in ["TransactionID", "isFraud"]]
X = df[features]
y = df["isFraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Supervised Models with progress bar
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "LightGBM": LGBMClassifier()
}

best_model = None
best_roc_auc = 0

for name, model in tqdm(models.items(), desc="Training Models"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"\n{name} Evaluation:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc)

    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        best_model = model

# Save the best model
joblib.dump(best_model, "best_fraud_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nBest model saved successfully!")

# Evaluate the best model on the test set
y_pred_best = best_model.predict(X_test)
y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]
print("\nBest Model Test Evaluation:")
print(classification_report(y_test, y_pred_best))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba_best))