# ============================================================
# File Name : 10-07-2026.py
# Topic : Renewable Energy Dataset Analysis
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# ============================================================
# Import Dataset
# ============================================================

df = pd.read_csv("complete_renewable_energy_dataset.csv")

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== LAST 5 ROWS ==========")
print(df.tail())

print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
print(df.columns)

print("\n========== INFO ==========")
print(df.info())

print("\n========== DESCRIPTION ==========")
print(df.describe(include="all"))

# ============================================================
# Generalization
# ============================================================

# This dataset contains renewable energy information.
# Every row represents one observation.
# Numerical columns contain measurable values.
# Categorical columns represent categories like country,
# energy source, region etc.
# Dataset can be used for sustainability analysis.

# ============================================================
# Relation Between Columns
# ============================================================

# Country → Renewable Energy Production
# Year → Renewable Energy Growth
# Population → Energy Consumption
# GDP → Energy Demand
# Carbon Emission ↔ Renewable Energy
# Renewable Energy increases → Carbon Emission decreases

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

print("\nAfter Handling Missing Values")
print(df.isnull().sum())

# ============================================================
# Duplicate Rows
# ============================================================

print("\nDuplicate Rows :", df.duplicated().sum())

df = df.drop_duplicates()

# ============================================================
# Feature Engineering
# ============================================================

# Create Total Score using first two numerical columns

if len(num_cols) >= 2:
    df["Combined_Feature"] = df[num_cols[0]] + df[num_cols[1]]

# ============================================================
# Label Encoding
# ============================================================

encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# ============================================================
# Scaling
# ============================================================

scaler = StandardScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

# ============================================================
# Correlation
# ============================================================

print("\nCorrelation Matrix")

corr = df.corr(numeric_only=True)

print(corr)

# ============================================================
# Visualization
# ============================================================

# Heatmap

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Histogram

df.hist(figsize=(12,10))
plt.show()

# Boxplot

plt.figure(figsize=(12,6))
sns.boxplot(data=df[num_cols])
plt.xticks(rotation=45)
plt.show()

# Pairplot

sns.pairplot(df[num_cols])
plt.show()

# Scatter Plot

if len(num_cols)>=2:
    sns.scatterplot(x=df[num_cols[0]],y=df[num_cols[1]])
    plt.title("Scatter Plot")
    plt.show()

# Countplot

if len(cat_cols)>0:
    sns.countplot(x=df[cat_cols[0]])
    plt.xticks(rotation=90)
    plt.show()

# ============================================================
# Insights
# ============================================================

print("""
================ INSIGHTS ================

1. Dataset imported successfully.

2. Missing values handled.
   Numerical columns -> Mean
   Categorical columns -> Mode

3. Duplicate records removed.

4. Label Encoding used because machine learning
   algorithms require numerical values.

5. Standard Scaling used because numerical
   columns have different ranges.

6. Feature Engineering created a new feature
   using numerical columns.

7. Heatmap shows relationship among features.

8. Histogram shows distribution.

9. Boxplot detects outliers.

10. Pairplot shows relation among numerical features.

11. Scatter Plot shows correlation.

12. Countplot shows frequency of categorical values.

""")

# ============================================================
# Summary
# ============================================================

print("""
================ SUMMARY ================

✔ Dataset Imported

✔ Dataset Explored

✔ Generalization Written

✔ Relationship Explained

✔ Missing Values Handled

✔ Feature Engineering Done

✔ Duplicate Removed

✔ Label Encoding Applied

✔ Standard Scaling Applied

✔ Seaborn Visualization Done

✔ Insights Generated

Assignment Completed Successfully

=========================================
""")