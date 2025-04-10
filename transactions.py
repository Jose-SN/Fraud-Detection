import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

#import river
from river import compose, preprocessing, linear_model, metrics
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df_train = pd.read_csv("train_transaction.csv")
df_identity = pd.read_csv("train_identity.csv")
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

# Train Supervised Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Evaluation:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Train Unsupervised Models
print("\nUnsupervised Learning")
kmeans = KMeans(n_clusters=2, random_state=42).fit(X_train)
dbscan = DBSCAN().fit(X_train)

print("K-Means Cluster Labels:", np.unique(kmeans.labels_))
print("DBSCAN Cluster Labels:", np.unique(dbscan.labels_))

# Online Learning with River
model_river = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression()
)
metric = metrics.Accuracy()

for x, y in zip(X_train, y_train):
    model_river = model_river.learn_one(dict(enumerate(x)), y)
    y_pred = model_river.predict_one(dict(enumerate(x)))
    metric = metric.update(y, y_pred)

print("\nRiver Model Accuracy:", metric)

# Save model for deployment
joblib.dump(models["Random Forest"], "fraud_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")

# Load model for deployment
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")