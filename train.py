import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier

os.makedirs("models", exist_ok=True)

# Load Dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Target Encoding
df["Attrition"] = df["Attrition"].map(
    {"Yes": 1, "No": 0}
)

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

categorical_cols = X.select_dtypes(
    include=["object"]
).columns

numerical_cols = X.select_dtypes(
    exclude=["object"]
).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_cols
        )
    ],
    remainder="passthrough"
)

# XGBoost
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train
pipeline.fit(
    X_train,
    y_train
)

# Predict
preds = pipeline.predict(X_test)

probs = pipeline.predict_proba(X_test)[:,1]

print("Accuracy:",
      accuracy_score(y_test,preds))

print("Precision:",
      precision_score(y_test,preds))

print("Recall:",
      recall_score(y_test,preds))

print("F1:",
      f1_score(y_test,preds))

print("ROC AUC:",
      roc_auc_score(
          y_test,
          probs
      ))

joblib.dump(
    pipeline,
    "models/attrition_model.pkl"
)

print("Model Saved Successfully")