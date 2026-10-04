import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_housing_data(filepath):
    # 1. Load Data
    df = pd.read_csv(filepath)
    print("--- DATA INFO ---")
    print(df.info())
    print(df.head(20))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # 2. Scope definition
    target = 'price'
    features = ['sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms', 'floors', 'yr_built', 'condition']
    df = df[[target] + features].copy()
    print("Selected features:")
    print([target] + features)
    print("Selected features after preprocessing:")
    print(df.head())
    print(f"Output shape: {df.shape}")

    # 3. Data Cleaning
    print("\nBefore Data Cleaning Nulls:\n", df.isnull().sum())
    df.dropna(inplace=True)  # Imputation via deletion
    df = df[df['bedrooms'] < 33]  # Noise reduction
    print("\nAfter Data Cleaning Nulls:\n", df.isnull().sum())
    print("\nData Description:\n", df.describe())

    # 4. Data Transformation
    df['log_price'] = np.log1p(df['price'])
    df['log_sqft_lot'] = np.log1p(df['sqft_lot'])

    # --- Expanded Visualization (2x2 Grid) ---
    # Increased height to 10 to accommodate two rows
    plt.figure(figsize=(14, 10))

    # Original Price Plot (Row 1, Left)
    plt.subplot(2, 2, 1)
    sns.histplot(df['price'], bins=50, kde=True, color='royalblue')
    plt.title('Original Price Distribution (Right-Skewed)')
    plt.xlabel('Price ($)')
    plt.ylabel('Count')

    # Log-Transformed Price Plot (Row 1, Right)
    plt.subplot(2, 2, 2)
    sns.histplot(df['log_price'], bins=50, kde=True, color='seagreen')
    plt.title('Log-Transformed Price (Normalized)')
    plt.xlabel('Log Price')
    plt.ylabel('Count')

    # Original Sqft_Lot Plot (Row 2, Left)
    plt.subplot(2, 2, 3)
    sns.histplot(df['sqft_lot'], bins=50, kde=True, color='coral')
    plt.title('Original Lot Size (Right-Skewed)')
    plt.xlabel('Lot Size (sqft)')
    plt.ylabel('Count')

    # Log-Transformed Sqft_Lot Plot (Row 2, Right)
    plt.subplot(2, 2, 4)
    sns.histplot(df['log_sqft_lot'], bins=50, kde=True, color='purple')
    plt.title('Log-Transformed Lot Size (Normalized)')
    plt.xlabel('Log Lot Size')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()
    # -----------------------------------------------

    # 5. Feature Selection
    # Drop raw price (replaced by log_price) and weak predictor (sqft_lot / log_sqft_lot)
    df_final = df.drop(columns=['price', 'sqft_lot', 'log_sqft_lot'])

    # 6. Final Split
    X = df_final.drop(columns=['log_price'])
    y = df_final['log_price']

    return X, y


# Execute the function
X_ready, y_ready = preprocess_housing_data(
    'D:/OneDrive/Desktop/Degree/Semester 4/Machine Learning/Individual assignment/housing.csv')