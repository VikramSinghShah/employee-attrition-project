import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.makedirs(
    "reports",
    exist_ok=True
)

df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Attrition Chart

plt.figure(figsize=(8,5))

sns.countplot(
    x="Attrition",
    data=df,
    palette="Set2"
)

plt.title(
    "Employee Attrition Distribution"
)

plt.savefig(
    "reports/attrition_distribution.png",
    dpi=300
)

plt.close()

# OverTime

plt.figure(figsize=(8,5))

sns.countplot(
    x="OverTime",
    hue="Attrition",
    data=df
)

plt.title(
    "OverTime vs Attrition"
)

plt.savefig(
    "reports/overtime_attrition.png",
    dpi=300
)

plt.close()

# Monthly Income

plt.figure(figsize=(10,6))

sns.boxplot(
    x="Attrition",
    y="MonthlyIncome",
    data=df
)

plt.savefig(
    "reports/salary_attrition.png",
    dpi=300
)

plt.close()

print("EDA Charts Saved")