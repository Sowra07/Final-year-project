import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(data.head())

# Separate the features (X) and the target variable (y)
X = data.drop('Calculated_bandgap', axis=1)
y = data['Calculated_bandgap']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
correlation_lr, _ = pearsonr(y_test, y_pred_lr)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
correlation_rf, _ = pearsonr(y_test, y_pred_rf)

# Neural Network
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
y_pred_nn = model.predict(X_test)
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
correlation_nn, _ = pearsonr(y_test, y_pred_nn.flatten())

# Comparison of Results
print(f'Linear Regression RMSE: {rmse_lr}, Pearson Correlation: {correlation_lr}')
print(f'Random Forest RMSE: {rmse_rf}, Pearson Correlation: {correlation_rf}')
print(f'Neural Network RMSE: {rmse_nn}, Pearson Correlation: {correlation_nn}')

# Determine the best approach based on RMSE
rmse_values = {
    'Linear Regression': rmse_lr,
    'Random Forest': rmse_rf,
    'Neural Network': rmse_nn
}

best_approach = min(rmse_values, key=rmse_values.get)
print(f'The best approach is {best_approach} based on RMSE.')

# Plotting the results
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.6, color='b', label='Predicted')
plt.scatter(y_test, y_test, alpha=0.3, color='r', label='Measured')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Measured Bandgap')
plt.ylabel('Predicted Bandgap')
plt.title('Linear Regression')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='b', label='Predicted')
plt.scatter(y_test, y_test, alpha=0.3, color='r', label='Measured')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Measured Bandgap')
plt.ylabel('Predicted Bandgap')
plt.title('Random Forest')
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_nn, alpha=0.6, color='b', label='Predicted')
plt.scatter(y_test, y_test, alpha=0.3, color='r', label='Measured')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Measured Bandgap')
plt.ylabel('Predicted Bandgap')
plt.title('Neural Network')
plt.legend()

plt.tight_layout()
plt.show()

# Correlation matrices
corr_matrix_lr = np.corrcoef(y_test, y_pred_lr)
corr_matrix_rf = np.corrcoef(y_test, y_pred_rf)
corr_matrix_nn = np.corrcoef(y_test, y_pred_nn.flatten())

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.heatmap(corr_matrix_lr, annot=True, cmap='coolwarm', xticklabels=['Measured', 'Predicted'], yticklabels=['Measured', 'Predicted'])
plt.title('Correlation Matrix (Linear Regression)')

plt.subplot(1, 3, 2)
sns.heatmap(corr_matrix_rf, annot=True, cmap='coolwarm', xticklabels=['Measured', 'Predicted'], yticklabels=['Measured', 'Predicted'])
plt.title('Correlation Matrix (Random Forest)')

plt.subplot(1, 3, 3)
sns.heatmap(corr_matrix_nn, annot=True, cmap='coolwarm', xticklabels=['Measured', 'Predicted'], yticklabels=['Measured', 'Predicted'])
plt.title('Correlation Matrix (Neural Network)')

plt.tight_layout()
plt.show()

# Output the best approach
print(f'The best approach is {best_approach} based on RMSE.')
