import warnings
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Suppress the warning about feature names
warnings.filterwarnings("ignore", message="X does not have valid feature names")

df = pandas.read_csv("DTC_data.csv")

d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Go']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features)
plt.show()

# Assume you have a new data point for prediction, you can replace it with your actual data.
new_data_point = [[30, 5, 10, 1]]  # Example values for Age, Experience, Rank, and Nationality

# Predict the value for the new data point
predicted_value = dtree.predict(new_data_point)

print("Predicted value:", predicted_value)
