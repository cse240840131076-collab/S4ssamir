# ============================================================
# File Name : 11-07-2026_Linear_Regression.py
# Topic : Linear Regression on Renewable Energy Dataset
# ============================================================

# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("complete_renewable_energy_dataset.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nInformation")
print(df.info())

print("\nDescription")
print(df.describe(include="all"))

# ============================================================
# Generalization
# ============================================================

# Every row represents renewable energy information.
# Numerical columns contain measurable values.
# Categorical columns represent country, region etc.
# Dataset is useful for sustainability analysis.

# ============================================================
# Missing Values
# ============================================================

print("\nMissing Values")
print(df.isnull().sum())

# Numerical columns

num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Categorical columns

cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# ============================================================
# Remove Duplicate Rows
# ============================================================

df = df.drop_duplicates()

# ============================================================
# Feature Engineering
# ============================================================

if len(num_cols) >= 2:
    df["Combined_Feature"] = df[num_cols[0]] + df[num_cols[1]]

# ============================================================
# Label Encoding
# ============================================================

encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# ============================================================
# Correlation
# ============================================================

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# ============================================================
# Scaling
# ============================================================

scaler = StandardScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

# ============================================================
# Target Column
# ============================================================

# Change this column according to your dataset

target = "Renewable_Energy_Production"

# Features

X = df.drop(target, axis=1)

# Target

y = df[target]

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data :", X_train.shape)
print("Testing Data :", X_test.shape)

# ============================================================
# Linear Regression Model
# ============================================================

model = LinearRegression()

model.fit(X_train, y_train)

# ============================================================
# Prediction
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# Evaluation
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print("MAE :", mae)

print("MSE :", mse)

print("RMSE :", rmse)

print("R2 Score :", r2)

# ============================================================
# Scatter Plot
# ============================================================

plt.figure(figsize=(6,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted")

plt.show()

# ============================================================
# Insights
# ============================================================

print("""

==================== INSIGHTS ====================

1. Dataset imported successfully.

2. Missing values handled.

3. Duplicate rows removed.

4. Categorical columns encoded using LabelEncoder.

5. Numerical columns scaled using StandardScaler.

6. Train-Test Split = 80:20

7. Linear Regression model trained.

8. Predictions generated.

9. MAE shows average prediction error.

10. MSE penalizes larger errors.

11. RMSE gives error in original units.

12. R² Score shows model accuracy.

Higher R² Score means better model performance.

==================================================

""")