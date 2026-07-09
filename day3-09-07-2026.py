# ============================================================
# File Name : 09-07-2026.py
# Topic : Green Skills / Sustainability Dataset Analysis
# ============================================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Import Dataset
# ------------------------------------------------------------

# Replace the file name with your dataset name
df = pd.read_csv("complete_renewable_energy_dataset.csv")

# ------------------------------------------------------------
# Display Dataset
# ------------------------------------------------------------

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== LAST 5 ROWS ==========")
print(df.tail())

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe(include='all'))

print("\n========== COLUMN NAMES ==========")
print(df.columns)

print("\n========== SHAPE ==========")
print(df.shape)

# ============================================================
# General Observations
# ============================================================

# Observation 1:
# Each row represents one record in the sustainability dataset.

# Observation 2:
# Numerical columns contain measurable values such as emissions,
# renewable energy percentage, energy consumption, etc.

# Observation 3:
# Categorical columns may contain country names, regions,
# industries, sectors, or years.

# Observation 4:
# Dataset can be used for environmental analysis.

# Observation 5:
# It helps identify sustainability trends.

# ============================================================
# Relationship Between Features
# ============================================================

# Country ↔ Renewable Energy
# Different countries have different renewable energy usage.

# Renewable Energy ↔ Carbon Emission
# Higher renewable energy generally reduces carbon emissions.

# Population ↔ Energy Consumption
# Larger populations usually consume more energy.

# GDP ↔ Energy Consumption
# High GDP countries generally consume more energy.

# GDP ↔ Carbon Emission
# Industrial growth can increase emissions.

# Year ↔ Renewable Energy
# Renewable energy generally increases over time.

# Year ↔ Carbon Emission
# Emissions may decrease if renewable energy adoption increases.

# ============================================================
# Missing Values
# ============================================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Fill numerical null values with mean

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Fill categorical null values with mode

categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n========== AFTER CLEANING ==========")
print(df.isnull().sum())

# ============================================================
# Duplicate Records
# ============================================================

print("\nDuplicate Rows =", df.duplicated().sum())

df = df.drop_duplicates()

# ============================================================
# Correlation
# ============================================================

print("\n========== CORRELATION MATRIX ==========")
print(df.corr(numeric_only=True))

# ============================================================
# Basic Analysis
# ============================================================

print("\n========== BASIC ANALYSIS ==========")

for col in numeric_cols:
    print(f"\nColumn : {col}")
    print("Mean :", df[col].mean())
    print("Maximum :", df[col].max())
    print("Minimum :", df[col].min())
    print("Median :", df[col].median())

# ============================================================
# Unique Values
# ============================================================

print("\n========== UNIQUE VALUES ==========")

for col in categorical_cols:
    print(col, ":", df[col].nunique())

# ============================================================
# Data Types
# ============================================================

print("\n========== DATA TYPES ==========")
print(df.dtypes)

# ============================================================
# Clean Dataset Preview
# ============================================================

print("\n========== CLEAN DATA ==========")
print(df.head())

# ============================================================
# Summary
# ============================================================

print("""
==================================================
SUMMARY
==================================================

1. Imported Green Skills / Sustainability dataset.

2. Displayed dataset using:
   - head()
   - tail()
   - info()
   - describe()

3. Observed dataset structure.

4. Explained relationships among important features.

5. Checked missing values.

6. Filled numerical missing values using Mean.

7. Filled categorical missing values using Mode.

8. Removed duplicate records.

9. Generated correlation matrix.

10. Calculated Mean, Maximum, Minimum and Median.

11. Displayed unique values and data types.

12. Dataset is cleaned and ready for visualization
    or machine learning.

==================================================
Assignment Completed Successfully
==================================================
""")