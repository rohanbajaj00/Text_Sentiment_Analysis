import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(42)
num_units = 50
num_obs_per_unit = 20

X = np.random.rand(num_units * num_obs_per_unit, 1)
individual_effects = np.random.normal(0, 0.5, size=(num_units, 1))
noise = np.random.normal(0, 0.2, size=(num_units * num_obs_per_unit, 1))
y = 2 * X + individual_effects.repeat(num_obs_per_unit, axis=0) + noise

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the residuals
residuals = y - model.predict(X)

# Create a residual plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X.flatten(), y=residuals.flatten(), hue=np.repeat(np.arange(num_units), num_obs_per_unit), palette="viridis", alpha=0.7)
plt.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero Residuals')
plt.title('Residual Plot with Individual Unit Effects')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.legend(title='Unit')
plt.show()
