import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Dataset
df_raw = pd.read_csv('D:/OneDrive/Desktop/Degree/Semester 4/Machine Learning/Individual assignment/housing.csv')

# Define your exact project scope
identifiers = ['id', 'date']
input_features = ['sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms', 'floors', 'yr_built', 'condition']
target = ['price']

# Create the working dataframe with all necessary columns
df = df_raw[identifiers + target + input_features].copy()

# Create a list of numerical columns for statistical math (strictly excluding id and date)
analysis_cols = target + input_features

print("=== TAILORED DATA QUALITY ANALYSIS ===")
print(f"Analyzing {len(df)} records across {len(df.columns)} selected attributes.\n")

# 2. Missing Values
missing = df.isnull().sum()
missing_df = pd.DataFrame({'Missing Count': missing})
print("--- MISSING VALUES ---")
print(missing_df[missing_df['Missing Count'] > 0])

# 3. Duplicate Records
# Check for exact identical rows across all columns
exact_duplicates = df.duplicated().sum()
# Check for duplicate IDs (the same house sold multiple times on different dates)
id_duplicates = df.duplicated(subset=['id']).sum()
# Check for duplicate transactions (the same house recorded as sold on the exact same date)
transaction_duplicates = df.duplicated(subset=['id', 'date']).sum()

print("\n--- DUPLICATE RECORDS ---")
print(f"Exact row duplicates (Identical data): {exact_duplicates}")
print(f"Duplicate IDs (Same house sold on different dates - Resales): {id_duplicates}")
print(f"Duplicate Transactions (Same house sold on the EXACT same date - System Error): {transaction_duplicates}")

# 4. Noise & Inconsistency
print("\n--- NOISE & INCONSISTENCY ---")
noisy_bed = df[df['bedrooms'] == 33]
print("Identified Noisy Record (Likely Typo):")
print(noisy_bed)

# 5. Skewness (Calculated only on analysis columns, ignoring ID and Date)
skewness = df[analysis_cols].skew().sort_values(ascending=False)
print("\n--- SKEWNESS ---")
print(skewness)

# 6. Outliers Detection (IQR Method)
outlier_summary = []
for col in analysis_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
    outlier_summary.append({'Feature': col, 'Outliers': outlier_count, 'Percentage (%)': round((outlier_count/len(df))*100, 2)})

outlier_df = pd.DataFrame(outlier_summary).sort_values(by='Percentage (%)', ascending=False)
print("\n--- OUTLIERS DETECTION ---")
print(outlier_df)

# 7. Correlation with Target (Price)
print("\n--- CORRELATION MATRIX ---")
corr_matrix = df[analysis_cols].corr()
print(corr_matrix['price'].sort_values(ascending=False))

# ==========================================
# VISUALIZATIONS (GRAPHS)
# ==========================================

# Graph 1: Feature Distributions (Price & Sqft_living)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['price'], bins=50, kde=True, color='royalblue')
plt.title('Distribution of Price (Target)')
plt.xlabel('Price ($)')

plt.subplot(1, 2, 2)
sns.histplot(df['sqft_living'], bins=50, kde=True, color='seagreen')
plt.title('Distribution of Living Area (sqft_living)')
plt.xlabel('Square Feet')
plt.tight_layout()
plt.show()

# Graph 2: Outliers (Boxplots for top skewed variables)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['price'], color='lightblue')
plt.title('Price Boxplot (Outliers)')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['sqft_lot'], color='lightgreen')
plt.title('Lot Size Boxplot (Outliers)')
plt.tight_layout()
plt.show()

# Graph 3: Condition Distribution (Categorical/Ordinal Imbalance)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='condition', palette='viridis', legend=False)
plt.title('Distribution of Property Condition (Class Imbalance)')
plt.xlabel('Condition Grade (1-5)')
plt.ylabel('Count')
plt.show()

# Graph 4: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Heatmap (Project Specific Features)')
plt.tight_layout()
plt.show()