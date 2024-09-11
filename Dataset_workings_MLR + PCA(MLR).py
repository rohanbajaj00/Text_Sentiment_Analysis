import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
file_path = 'Similarity Results/pca_results_00_23.csv'
data = pd.read_csv(file_path, index_col='Date', parse_dates=True)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Function to create lagged variables
def create_lagged_features(df, lag):
    return df.shift(lag)

# Function to calculate the percentage of times actual and prediction are on the same side of 0
def calculate_same_side_percentage(actual, predicted):
    same_side = np.sign(actual) == np.sign(predicted)
    percentage = np.mean(same_side) * 100
    return percentage

# Find the best lag for each predictor based on the lowest MSE
lags = range(1, 11)
best_lags = {}

for col in train_data.columns:
    if col == 'FTSE_Returns':
        continue
    mse_scores = []
    for lag in lags:
        lagged_feature = create_lagged_features(train_data[col], lag)
        model = LinearRegression()
        model.fit(lagged_feature.dropna().values.reshape(-1, 1), train_data['FTSE_Returns'][lag:])
        predictions = model.predict(lagged_feature.dropna().values.reshape(-1, 1))
        mse = mean_squared_error(train_data['FTSE_Returns'][lag:], predictions)
        mse_scores.append(mse)
    best_lag = lags[np.argmin(mse_scores)]
    best_lags[col] = best_lag

# Create lagged features using the best lags
lagged_train_data = pd.DataFrame(index=train_data.index)
lagged_test_data = pd.DataFrame(index=test_data.index)

for col, lag in best_lags.items():
    lagged_train_data[col] = create_lagged_features(train_data[col], lag)
    lagged_test_data[col] = create_lagged_features(test_data[col], lag)

# Drop rows with NaN values created by lagging
lagged_train_data.dropna(inplace=True)
train_target = train_data['FTSE_Returns'].loc[lagged_train_data.index]

# Train the regression model
model = LinearRegression()
model.fit(lagged_train_data, train_target)

# Align the test target with the lagged test data
lagged_test_data.dropna(inplace=True)
test_target = test_data['FTSE_Returns'].loc[lagged_test_data.index]

# Predict on the testing data
predictions = model.predict(lagged_test_data)

# Calculate MSE
mse = mean_squared_error(test_target, predictions)
print(f"Mean Squared Error on Test Data: {mse}")

# Calculate the percentage of times actual and prediction are on the same side of 0
same_side_percentage = calculate_same_side_percentage(test_target, predictions)
print(f"Percentage of times actual and predicted are on the same side of 0: {same_side_percentage:.2f}%")

# Calculate the standard deviations
predictions_std = np.std(predictions)
test_target_std = np.std(test_target)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(test_target.index, test_target, label='Actual FTSE Returns')
plt.plot(test_target.index, predictions, label='Predicted FTSE Returns', linestyle='--')

# Add horizontal lines at ±2 standard deviations for actual FTSE returns
plt.axhline(y=2*test_target_std, color='green', linestyle=':', label='2 SD Actual')
plt.axhline(y=-2*test_target_std, color='green', linestyle=':')

# Add vertical lines where predicted values are above or below 1.5 standard deviations
outlier_indices = test_target.index[(predictions > 10*predictions_std) | (predictions < -1*predictions_std)]
for outlier in outlier_indices:
    plt.axvline(x=outlier, color='red', linestyle='--', alpha=0.5)

plt.legend()
plt.title('Actual vs Predicted FTSE Returns with SD Thresholds')
plt.xlabel('Date')
plt.ylabel('FTSE Returns')
plt.show()

# Save the data for comparison
comparison_df = pd.DataFrame({
    'Date': test_target.index,
    'Actual': test_target.values,
    'Predicted': predictions
})
comparison_df.to_csv('ftse_returns_comparison.csv', index=False)

print("Predictions saved to 'ftse_returns_comparison.csv'.")
