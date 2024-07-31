import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

# Load the dataset
dataset = pd.read_csv('some_data.csv')

# Separate features (x_vals) and target variable (y_vals)
x_vals = dataset.iloc[:, :-1]
y_vals = dataset.iloc[:, -1]

# Identify categorical columns for one-hot encoding
categorical_columns = ['State']  # Replace with the actual categorical column name

# Create a transformer for one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), categorical_columns)
    ],
    remainder='passthrough'
)

# Apply the transformer to the features
x_vals_encoded = preprocessor.fit_transform(x_vals)

# Display the one-hot encoded values
print("One-Hot Encoded Values:")
print(x_vals_encoded)

# Print a separator line
print("\n" + "-"*50 + "\n")

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_vals_encoded, y_vals, test_size=0.2, random_state=0)

# Fit the linear regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(x_test)

# Display the R-squared score with custom formatting
formatted_r_squared = "{:.12f}".format(r2_score(y_test, y_pred))
print("R-squared score (One-Hot Encoded):", formatted_r_squared)
