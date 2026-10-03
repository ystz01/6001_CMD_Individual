import pandas as pd

# 1. Load the dataset
df = pd.read_csv('D:/OneDrive/Desktop/Degree/Semester 4/Machine Learning/Individual assignment/housing.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 2. Define your specific input variables (and target if desired)
input_variables = [
    'sqft_living',
    'sqft_lot',
    'bedrooms',
    'bathrooms',
    'floors',
    'yr_built',
    'condition',
    'price' # Optional: include your target variable to see its stats too
]

# 3. Generate the summary statistics table
# .describe() automatically calculates count, mean, std, min, Q1, Median, Q3, and max
summary_table = df[input_variables].describe().round(2)

# 4. Display the table
print(summary_table)


# Optional: If you are using a Jupyter Notebook, just typing `summary_table` at the end
# of the cell will render it with the exact visual formatting seen in your reference image.
# summary_table