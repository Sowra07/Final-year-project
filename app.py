# app.py

from flask import Flask, render_template, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for handling cross-origin requests

# Load your dataset
@app.route('/')
def root():
    return render_template('index.html')

@app.route('/load_data')
def load_data():
    data = pd.read_csv('data.csv')
    return data.to_json()

# Linear Regression endpoint
@app.route('/linear_regression')
def linear_regression():
    # Load data
    data = pd.read_csv('data.csv')
    X = data.drop('Calculated_bandgap', axis=1)
    y = data['Calculated_bandgap']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Make predictions
    y_pred = lr.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Prepare HTML response
    return render_template('linear_regression.html', rmse=rmse)

    return html_response

# Random Forest endpoint
@app.route('/random_forest')
def random_forest():
    # Load data
    data = pd.read_csv('data.csv')
    X = data.drop('Calculated_bandgap', axis=1)
    y = data['Calculated_bandgap']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Prepare HTML response

    return render_template('random_forest.html', rmse=rmse)

# Neural Network endpoint
@app.route('/neural_network')
def neural_network():
    # Load data
    data = pd.read_csv('data.csv')
    X = data.drop('Calculated_bandgap', axis=1)
    y = data['Calculated_bandgap']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train model
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)

    # Make predictions
    y_pred = model.predict(X_test_scaled).flatten()

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Prepare HTML response
    

    return render_template('neural_network.html', rmse=rmse)

# Comparison endpoint
@app.route('/comparison')
def comparison():
    # Implement comparison logic here
    # For simplicity, you can just return a placeholder response
    return render_template('comparison.html')

if __name__ == '__main__':
    app.run(debug=True)  # Run the application in debug mode for development
