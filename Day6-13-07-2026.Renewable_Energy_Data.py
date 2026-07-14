# ==========================================================
# File Name : 12-07-2026_KMeans_Clustering.py
# Topic : K-Means Clustering on Renewable Energy Dataset
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("Renewable_Energy_Data.csv")

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== INFO ==========")
print(df.info())

print("\n========== DESCRIPTION ==========")
print(df.describe(include="all"))

# ==========================================================
# Generalization
# ==========================================================

# Region -> Geographic area
# Energy_Source -> Solar, Wind, Hydro etc.
# Temperature_C -> Temperature
# Wind_Speed_m_s -> Wind speed
# Solar_Radiation_kWh_m2 -> Solar radiation
# Rainfall_mm -> Rainfall
# Season -> Season of data
# Efficiency_Ratio -> Renewable energy efficiency
# Lagged_Production_MWh -> Previous production
# Combined_Weather_Index -> Weather indicator
# Energy_Class -> Low / Medium / High production category

# ==========================================================
# Handle Missing Values
# ==========================================================

print("\nMissing Values")
print(df.isnull().sum())

num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# ==========================================================
# Remove Duplicate Rows
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
# Label Encoding
# ==========================================================

encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# ==========================================================
# Correlation Heatmap
# ==========================================================

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ==========================================================
# Histogram
# ==========================================================

df.hist(figsize=(12,10))
plt.show()

# ==========================================================
# Boxplot
# ==========================================================

plt.figure(figsize=(12,6))
sns.boxplot(data=df[num_cols])
plt.xticks(rotation=45)
plt.show()

# ==========================================================
# Scaling
# ==========================================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

# ==========================================================
# Elbow Method
# ==========================================================

wcss = []

for i in range(1,11):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(scaled_data)

    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))

plt.plot(range(1,11), wcss, marker="o")

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.show()

# ==========================================================
# KMeans Model
# ==========================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(scaled_data)

df["Cluster"] = clusters

# ==========================================================
# Cluster Distribution
# ==========================================================

print("\nCluster Count")

print(df["Cluster"].value_counts())

# ==========================================================
# Scatter Plot
# ==========================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=df["Temperature_C"],
    y=df["Efficiency_Ratio"],
    hue=df["Cluster"],
    palette="Set1"
)

plt.title("K-Means Clustering")

plt.show()

# ==========================================================
# Evaluation
# ==========================================================

score = silhouette_score(scaled_data, clusters)

print("\nSilhouette Score")

print(score)

# ==========================================================
# Insights
# ==========================================================

print("""

==================== INSIGHTS ====================

1. Imported Renewable Energy dataset.

2. Cleaned missing values.

3. Removed duplicate records.

4. Encoded categorical columns.

5. Created Weather_Score feature.

6. StandardScaler used because K-Means
   works better on scaled data.

7. Elbow Method used to find optimal K.

8. Applied K-Means Clustering.

9. Silhouette Score used for evaluation.

10. Higher Silhouette Score means
    better cluster separation.

===================================================

""")

# ==========================================================
# Summary
# ==========================================================

print("""

================ SUMMARY ================

✔ Dataset Imported

✔ Dataset Analysed

✔ Missing Values Handled

✔ Feature Engineering Done

✔ Label Encoding Applied

✔ Scaling Applied

✔ Seaborn Visualizations Completed

✔ Elbow Method Applied

✔ K-Means Clustering Applied

✔ Silhouette Score Calculated

✔ Insights Generated

Assignment Completed Successfully

=========================================

""")