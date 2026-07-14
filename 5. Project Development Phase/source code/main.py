# ============================================================
# CREDIT CARD APPROVAL PREDICTION USING MACHINE LEARNING
# ============================================================

import os
import pickle
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# ============================================================
# Create Folder for Saving Graphs
# ============================================================

os.makedirs("plots", exist_ok=True)

# ============================================================
# Graph Theme
# ============================================================

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ============================================================
# DATA COLLECTION
# ============================================================

print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

application = pd.read_csv("application_record.csv")
credit = pd.read_csv("credit_record.csv")

print("\nApplication Dataset")
print(application.head())

print("\nCredit Dataset")
print(credit.head())

print("\nApplication Shape :", application.shape)
print("Credit Shape      :", credit.shape)

# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n")
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(application.info())

print("\nMissing Values")
print(application.isnull().sum())

print("\nDescriptive Statistics")
print(application.describe())

# ============================================================
# UNIVARIATE ANALYSIS
# ============================================================

print("\nOccupation Type Count")
print(application["OCCUPATION_TYPE"].value_counts())
print("Generating Occupation Distribution...")
plt.figure(figsize=(14,6))
sns.countplot(
    x="OCCUPATION_TYPE",
    data=application,
    order=application["OCCUPATION_TYPE"].value_counts().index
)
plt.xticks(rotation=90)
plt.title("Occupation Distribution")
plt.tight_layout()
plt.savefig("plots/occupation_distribution.png")
plt.show()
print("Occupation Distribution Saved")
# ============================================================
# Gender Distribution
# ============================================================
print("Generating Occupation Distribution...")
plt.figure(figsize=(6,5))

sns.countplot(
    x="CODE_GENDER",
    data=application,
    palette="Set2"
)

plt.title("Gender Distribution")
plt.savefig("plots/gender_distribution.png")
plt.show()
print("Occupation Distribution Saved")
# ============================================================
# Income Distribution
# ============================================================
print("Generating Occupation Distribution...")
plt.figure(figsize=(10,5))

sns.histplot(
    application["AMT_INCOME_TOTAL"],
    bins=40,
    kde=True
)

plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Count")

plt.savefig("plots/income_distribution.png")
plt.show()
print("Occupation Distribution Saved")
# ============================================================
# Income Type
# ============================================================
print("Generating Occupation Distribution...")
plt.figure(figsize=(10,5))

sns.countplot(
    x="NAME_INCOME_TYPE",
    data=application,
    order=application["NAME_INCOME_TYPE"].value_counts().index
)

plt.xticks(rotation=30)
plt.title("Income Type Distribution")

plt.savefig("plots/income_type.png")
plt.show()
print("Occupation Distribution Saved")
# ============================================================
# DATA PREPROCESSING
# ============================================================

print("\n")
print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Remove duplicate applicants

application.drop_duplicates(inplace=True)

# Fill Missing Occupation

application["OCCUPATION_TYPE"] = application[
    "OCCUPATION_TYPE"
].fillna("Unknown")

print("\nMissing Values After Cleaning")

print(application.isnull().sum())

# ============================================================
# CREATE TARGET VARIABLE
# ============================================================

credit["TARGET"] = credit["STATUS"].apply(
    lambda x: 1 if x in ["1","2","3","4","5"] else 0
)

credit = credit.groupby("ID")["TARGET"].max().reset_index()

print("\nCredit Dataset After Grouping")

print(credit.head())

# ============================================================
# MERGE DATASETS
# ============================================================

data = pd.merge(
    application,
    credit,
    on="ID",
    how="inner"
)

print("\nMerged Dataset Shape :", data.shape)

print(data.head())

# ============================================================
# TARGET DISTRIBUTION
# ============================================================
print("Generating Occupation Distribution...")
plt.figure(figsize=(6,5))

sns.countplot(
    x="TARGET",
    data=data,
    palette="viridis"
)

plt.title("Target Distribution")

plt.savefig("plots/target_distribution.png")

plt.show()
print("Occupation Distribution Saved")
print("\nTarget Distribution")

print(data["TARGET"].value_counts())

# ============================================================
# LABEL ENCODING
# ============================================================

categorical_columns = [

    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE"

]

print("\n" + "=" * 60)
print("LABEL ENCODING")
print("=" * 60)

# Dictionary to store encoders
label_encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    data[column] = encoder.fit_transform(data[column].astype(str))

    label_encoders[column] = encoder

print("\nEncoded Dataset")
print(data.head())

# ============================================================
# LABEL ENCODING MAPPINGS
# ============================================================

print("\n" + "=" * 60)
print("LABEL ENCODING MAPPINGS")
print("=" * 60)

for column, encoder in label_encoders.items():

    print(f"\n{column}")
    print("-" * 40)

    for index, value in enumerate(encoder.classes_):

        print(f"{index} --> {value}")

# ============================================================
# CORRELATION HEATMAP
# ============================================================

print("\nGenerating Correlation Heatmap...")

correlation = data.corr(numeric_only=True)

plt.figure(figsize=(15,10))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("plots/correlation_heatmap.png")

plt.close()

print("Correlation Heatmap Saved Successfully.")
# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X = data.drop(columns=["ID","TARGET"])

y = data["TARGET"]

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Shape :", X_train.shape)

print("Testing Shape  :", X_test.shape)

print("\nTraining Target Distribution")

print(y_train.value_counts())

print("\nTesting Target Distribution")

print(y_test.value_counts())

print("\n")

print("=" * 60)

print("MODEL BUILDING")

print("=" * 60)
# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print(f"Accuracy : {lr_accuracy:.4f}")
print(f"F1 Score : {lr_f1:.4f}")

print("\nConfusion Matrix")
cm_lr = confusion_matrix(y_test, lr_pred)
print(cm_lr)

print("\nClassification Report")
print(classification_report(y_test, lr_pred))
print("Generating Occupation Distribution...")
plt.figure(figsize=(6,5))
sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/logistic_confusion_matrix.png")
plt.show()
print("Occupation Distribution Saved")

# ============================================================
# DECISION TREE
# ============================================================

print("\n" + "=" * 60)
print("DECISION TREE")
print("=" * 60)

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)

print(f"Accuracy : {dt_accuracy:.4f}")
print(f"F1 Score : {dt_f1:.4f}")

print("\nConfusion Matrix")
cm_dt = confusion_matrix(y_test, dt_pred)
print(cm_dt)

print("\nClassification Report")
print(classification_report(y_test, dt_pred))
print("Generating Occupation Distribution...")
plt.figure(figsize=(6,5))
sns.heatmap(
    cm_dt,
    annot=True,
    fmt="d",
    cmap="Greens"
)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/decision_tree_confusion_matrix.png")
plt.show()
print("Occupation Distribution Saved")

# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print(f"Accuracy : {rf_accuracy:.4f}")
print(f"F1 Score : {rf_f1:.4f}")

print("\nConfusion Matrix")
cm_rf = confusion_matrix(y_test, rf_pred)
print(cm_rf)

print("\nClassification Report")
print(classification_report(y_test, rf_pred))
print("Generating Occupation Distribution...")
plt.figure(figsize=(6,5))
sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Oranges"
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/random_forest_confusion_matrix.png")
plt.show()
print("Occupation Distribution Saved")

# ============================================================
# MODEL COMPARISON
# ============================================================

results = pd.DataFrame({

    "Model":[
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy":[
        lr_accuracy,
        dt_accuracy,
        rf_accuracy
    ],

    "F1 Score":[
        lr_f1,
        dt_f1,
        rf_f1
    ]

})

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results)

# ============================================================
# MODEL COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(10,6))

sns.barplot(
    data=results,
    x="Model",
    y="Accuracy",
    palette="Set2"
)

plt.title("Model Accuracy Comparison")
plt.ylim(0,1)

for i, value in enumerate(results["Accuracy"]):
    plt.text(i, value+0.01, f"{value:.3f}", ha="center")

plt.tight_layout()
plt.savefig("plots/model_accuracy_comparison.png")
plt.show()
print("Occupation Distribution Saved")

# ============================================================
# F1 SCORE GRAPH
# ============================================================
print("Generating Occupation Distribution...")
plt.figure(figsize=(10,6))

sns.barplot(
    data=results,
    x="Model",
    y="F1 Score",
    palette="viridis"
)

plt.title("Model F1 Score Comparison")
plt.ylim(0,1)

for i, value in enumerate(results["F1 Score"]):
    plt.text(i, value+0.01, f"{value:.3f}", ha="center")

plt.tight_layout()
plt.savefig("plots/model_f1_comparison.png")
plt.show()
print("Occupation Distribution Saved")

# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature":X.columns,
    "Importance":rf_model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Important Features")

print(importance.head(10))
print("Generating Occupation Distribution...")
plt.figure(figsize=(12,7))

sns.barplot(

    data=importance.head(10),

    x="Importance",

    y="Feature",

    palette="rocket"

)

plt.title("Top 10 Important Features")

plt.tight_layout()

plt.savefig("plots/feature_importance.png")

plt.show()
print("Occupation Distribution Saved")

# ============================================================
# SAVE BEST MODEL
# ============================================================

best_model = rf_model

pickle.dump(
    best_model,
    open("model.pkl","wb")
)

print("\nBest Model Saved Successfully as model.pkl")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL SUMMARY")
print("=" * 60)

print(f"""
Logistic Regression
Accuracy : {lr_accuracy:.4f}
F1 Score : {lr_f1:.4f}

Decision Tree
Accuracy : {dt_accuracy:.4f}
F1 Score : {dt_f1:.4f}

Random Forest
Accuracy : {rf_accuracy:.4f}
F1 Score : {rf_f1:.4f}
""")

best_index = results["F1 Score"].idxmax()

print(
    "Best Performing Model :",
    results.loc[best_index, "Model"]
)

print(
    "Best F1 Score :",
    round(results.loc[best_index, "F1 Score"],4)
)

print("\nProject Completed Successfully.")
print("All graphs have been saved inside the 'plots' folder.")
print("Machine Learning model saved as 'model.pkl'.")