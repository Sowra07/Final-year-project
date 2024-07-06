import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to calculate Pearson correlation coefficient
def pearson_corr(x, y):
    return np.corrcoef(x, y)[0, 1]

# Load the data from CSV file (user input)
file_path = 'data.csv'.strip()
data = pd.read_csv(file_path)

# Separate the features (X) and the target variable (y)
X = data.drop('Calculated_bandgap', axis=1)  # Assuming 'Calculated_bandgap' is the target variable
y = data['Calculated_bandgap']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Neural Network': tf.keras.Sequential([
                        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                        tf.keras.layers.Dense(32, activation='relu'),
                        tf.keras.layers.Dense(1)
                      ])
}

# Standardize data for Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate models
results = {}
for name, model in models.items():
    if name == 'Neural Network':
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
        y_pred = model.predict(X_test_scaled).flatten()
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pearson_corr_val = pearson_corr(y_test, y_pred)
    results[name] = {'RMSE': rmse, 'Pearson Correlation': pearson_corr_val}

# Compare results
for name, result in results.items():
    print(f'{name} - RMSE: {result["RMSE"]:.2f}, Pearson Correlation: {result["Pearson Correlation"]:.2f}')

# Plotting the results
plt.figure(figsize=(10, 6))
for name, model in models.items():
    if name == 'Neural Network':
        y_pred = model.predict(X_test_scaled).flatten()
    else:
        y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred, alpha=0.6, label=name)
plt.plot([min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())], 
         [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())], 'k--', lw=2)
plt.xlabel('Measured Bandgap')
plt.ylabel('Predicted Bandgap')
plt.title('Comparison: Measured vs Predicted Bandgap')
plt.legend()
plt.show()

# Plot correlation matrices
plt.figure(figsize=(15, 5))
for i, (name, model) in enumerate(models.items()):
    plt.subplot(1, len(models), i+1)
    if name == 'Neural Network':
        y_pred = model.predict(X_test_scaled).flatten()
    else:
        y_pred = model.predict(X_test)
    corr_matrix = np.corrcoef(y_test, y_pred)
    sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f')
    plt.title(f'{name}: Correlation Matrix')
plt.tight_layout()
plt.show()

# Determine the best approach
best_approach = min(results, key=lambda x: results[x]['RMSE'])
print(f'The best approach is {best_approach} based on RMSE.')
