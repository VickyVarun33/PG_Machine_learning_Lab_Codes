import pandas as pd

# Define the dataset
data = {
    'Weather': ['Rainy', 'Sunny', 'Overcast', 'Overcast', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Rainy', 'Sunny', 'Sunny', 'Rainy', 'Overcast', 'Overcast'],
    'Play': ['Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

# Convert to dataframe
df = pd.DataFrame(data)

# Function to generate frequency table
def generate_frequency_table(df, feature, target):
    return df.groupby([feature, target]).size().unstack(fill_value=0)

# Generate frequency table for Weather and Play
weather_play_freq = generate_frequency_table(df, 'Weather', 'Play')

# Calculate likelihood table
likelihood_table = weather_play_freq.apply(lambda x: x / x.sum(), axis=1)

# Calculate prior probability
prior_prob_yes = df['Play'].value_counts(normalize=True)['Yes']

# Calculate P(Yes|Sunny) and P(No|Sunny) using Bayes' theorem
p_sunny_given_yes = likelihood_table.loc['Sunny', 'Yes']
p_sunny_given_no = likelihood_table.loc['Sunny', 'No']

p_yes = prior_prob_yes
p_no = 1 - prior_prob_yes

p_yes_given_sunny = (p_sunny_given_yes * p_yes) / likelihood_table.loc['Sunny'].sum()
p_no_given_sunny = (p_sunny_given_no * p_no) / likelihood_table.loc['Sunny'].sum()

# Display performance of each step
print("Performance of each step:")
print(f"Likelihood:\n{likelihood_table}")
print(f"Evidence (prior probability of 'Yes'): {prior_prob_yes}")
print(f"Prior probability of 'Yes': {p_yes}")
print(f"Posterior probability of 'Yes' given Sunny weather: {p_yes_given_sunny}")
print(f"Posterior probability of 'No' given Sunny weather: {p_no_given_sunny}")

# Get user input for test condition
test_condition = input("Enter the test condition (e.g., 'Sunny' or 'Rainy'): ").capitalize()

# Determine whether to play or not based on posterior probabilities
if test_condition == 'Sunny':
    decision = 'play' if p_yes_given_sunny > p_no_given_sunny else 'not play'
    print(f"On a {test_condition} day, the player should {decision} the game.")
elif test_condition == 'Rainy':
    decision = 'play' if (1 - p_yes_given_sunny) > (1 - p_no_given_sunny) else 'not play'
    print(f"On a {test_condition} day, the player should {decision} the game.")
else:
    print("Invalid test condition. Please enter either 'Sunny' or 'Rainy'.")
