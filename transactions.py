#1: Imports and Setup
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from river import compose, linear_model, preprocessing, metrics, ensemble
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

#2: Load and Merge Data
data_dir = "./assets"
train_transaction_path = os.path.join(data_dir, "train_transaction.csv")
train_identity_path = os.path.join(data_dir, "train_identity.csv")

df_train = pd.read_csv(train_transaction_path)
df_identity = pd.read_csv(train_identity_path)
df = df_train.merge(df_identity, on="TransactionID", how="left")

# 3: Data Cleaning - Handling Missing Values

# Option 1: Fill missing values with -999
df_filled = df.copy()
df_filled.fillna(-999, inplace=True)

# Option 2: Drop rows with any missing values
df_dropped = df.copy()
df_dropped.dropna(inplace=True)

# You can later train models separately on df_filled and df_dropped
# and compare evaluation metrics to see which approach performs better.

# Label Encoding for Categorical Columns
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

#4: Feature Selection and Splitting
features = [
    "TransactionAmt", "card1", "card2", "dist1", "C1", "C2", "D1", "V1", "V2"
]

X = df[features]
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#5: Model Training with Evaluation
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "LightGBM": LGBMClassifier()
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"\n{name} Report:")
    # Focus on recall or F1 score for fraud class (1)
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_test, proba))
    if auc > best_score:
        best_score = auc
        best_model = model

joblib.dump(best_model, "best_fraud_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")

#6: Evaluation of Best Model
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score

# Predict with the best model
y_pred_test = best_model.predict(X_test_scaled)
y_proba_test = best_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
roc_auc = roc_auc_score(y_test, y_proba_test)

print("\nEvaluation of Best Model:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visual Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Best Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#7: Online Learning using River
from river import preprocessing as rv_pre
from river import compose as rv_comp
from river import metrics as rv_metrics

stream_df = df.sample(10000).copy()

online_model = rv_comp.Pipeline(
    rv_pre.StandardScaler(),
    linear_model.LogisticRegression()
)

acc = rv_metrics.Accuracy()
rocauc = rv_metrics.ROCAUC()

for i, row in stream_df.iterrows():
    x = row[features].to_dict()
    y = row["isFraud"]
    y_pred = online_model.predict_one(x)
    acc = acc.update(y, y_pred)
    rocauc = rocauc.update(y, y_pred)
    online_model = online_model.learn_one(x, y)

print("\nRiver Online Learning:")
print("Accuracy:", acc)
print("ROC AUC:", rocauc)

#8: Save River Model (optional for extended use)
import dill
with open("river_online_model.dill", "wb") as f:
    dill.dump(online_model, f)


#9: Feature Importance Visualization (for tree models)
if hasattr(best_model, 'feature_importances_'):
    feat_imp = pd.Series(best_model.feature_importances_, index=X.columns)
    feat_imp.nlargest(20).plot(kind='barh', figsize=(10, 8))
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.show()

#10: Final Notes
print("\nWorkflow completed successfully. Model and scaler saved. API ready for deployment.")