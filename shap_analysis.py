import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

os.makedirs("reports", exist_ok=True)

df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

df["Attrition"] = df["Attrition"].map(
    {"Yes":1, "No":0}
)

X = df.drop(
    "Attrition",
    axis=1
)

pipeline = joblib.load(
    "models/attrition_model.pkl"
)

preprocessor = pipeline.named_steps[
    "preprocessor"
]

model = pipeline.named_steps[
    "model"
]

X_transformed = preprocessor.transform(X)

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_transformed
)

plt.figure()

shap.summary_plot(
    shap_values,
    X_transformed,
    show=False
)

plt.savefig(
    "reports/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

print(
 "SHAP Summary Saved"
)