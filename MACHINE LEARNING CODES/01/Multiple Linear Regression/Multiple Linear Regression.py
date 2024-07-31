import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Assuming some_data.csv contains your data
dataset = pd.read_csv('some_data.csv')
x_vals = pd.get_dummies(dataset.iloc[:, :-1], drop_first=True)
y_vals = dataset.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print("R-squared score:", r2_score(y_test, y_pred))
# Plotting
plt.scatter(y_test, y_pred, color='blue')
plt.plot(plt.xlim(), plt.xlim(), color='red')  # Plotting the regression line
plt.title('Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


