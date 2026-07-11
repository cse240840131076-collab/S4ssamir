# ==========================================================
# File Name : 11-07-2026_Logistic_Regression.py
# Renewable Energy Dataset - Logistic Regression
# ==========================================================

# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("renewable_energy_data.csv")

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== INFO ==========")
print(df.info())

print("\n========== DESCRIPTION ==========")
print(df.describe(include="all"))

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# ==========================================================
# Generalization
# ==========================================================

# Region -> Geographic location
# Energy_Source -> Solar, Wind, Hydro etc.
# Temperature_C -> Temperature
# Wind_Speed_m_s -> Wind Speed
# Solar_Radiation_kWh_m2 -> Solar Radiation
# Rainfall_mm -> Rainfall
# Season -> Summer, Winter etc.
# Efficiency_Ratio -> Energy efficiency
# Lagged_Production_MWh -> Previous energy production
# Combined_Weather_Index -> Weather index
# Energy_Class -> Target Class (Low, Medium, High)

# ==========================================================
# Handle Missing Values
# ==========================================================

num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ==========================================================
# Remove Duplicates
# ==========================================================

df.drop_duplicates(inplace=True)

# ==========================================================
# Feature Engineering
# ==========================================================

df["Weather_Score"] = (
    df["Temperature_C"]
    + df["Wind_Speed_m_s"]
    + df["Solar_Radiation_kWh_m2"]
)

# ==========================================================
# Visualization
# ==========================================================

sns.countplot(x="Energy_Class", data=df)
plt.title("Energy Class Distribution")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df)
plt.show()

# ==========================================================
# Label Encoding
# ==========================================================

encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# ==========================================================
# Scaling
# ==========================================================

X = df.drop("Energy_Class", axis=1)
y = df["Energy_Class"]

scaler = StandardScaler()

X = scaler.fit_transform(X)

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# ==========================================================
# Logistic Regression Model
# ==========================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ==========================================================
# Prediction
# ==========================================================

y_pred = model.predict(X_test)

# ==========================================================
# Evaluation
# ==========================================================

print("\nAccuracy")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================================================
# Confusion Matrix Plot
# ==========================================================

plt.figure(figsize=(6,5))

sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    cmap="Blues",
    fmt="d"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# ==========================================================
# Insights
# ==========================================================

print("""

================ INSIGHTS =================

1. Dataset imported successfully.

2. Missing values filled.

3. Duplicate rows removed.

4. Weather_Score feature created.

5. Label Encoding converts text into numbers.

6. StandardScaler normalizes numerical values.

7. Logistic Regression is used because Energy_Class
   is a categorical variable.

8. Dataset split into 80% training and 20% testing.

9. Accuracy measures overall prediction performance.

10. Confusion Matrix shows correct and incorrect predictions.

11. Classification Report gives Precision,
    Recall and F1-Score.

===========================================

""")

# ==========================================================
# Summary
# ==========================================================

print("""

================ SUMMARY ================

✔ Dataset Imported

✔ Dataset Analysed

✔ Generalization Written

✔ Null Values Handled

✔ Feature Engineering Done

✔ Seaborn Visualizations Completed

✔ Label Encoding Applied

✔ Standard Scaling Applied

✔ Train Test Split Completed

✔ Logistic Regression Applied

✔ Model Evaluated

✔ Insights Generated

Assignment Completed Successfully

=========================================

""")