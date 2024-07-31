from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Your custom dataset
data = {
    'HAIR': [1110, 5000, 74580, 80000, 70000, 100000, 200000],  # Assuming 'HAIR' is the count of hairs, X1 Feature
    'FEATHER': [10, 119, 30, 51,50, 91, 80],  # Add other features here, X2 Feature
    'Label': ['Fish', 'Fish', 'Fish', 'PIG', 'PIG', 'Fish', 'PIG'], # This is the Target Variable.
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['HAIR', 'FEATHER']]
y = df['Label']

# Choose a classifier - K-Nearest Neighbors in this case
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the classifier on the entire dataset
knn_classifier.fit(X, y)

# Get input from the user for a new sample
hair = float(input("Enter the Animal hair Color: "))
feather = float(input("Enter the Feather Color: "))

# Make prediction for the new sample
K_Value = [[hair, feather]]

# Suppress the warning regarding feature names
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

prediction = knn_classifier.predict(K_Value)

print(f"The predicted category for the input is: {prediction[0]}")
