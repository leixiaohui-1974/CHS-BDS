import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

class RainfallPredictor:
    """
    A class to predict rainfall based on PWV time series data.
    """

    def __init__(self, timesteps, pwv_data, rainfall_data):
        """
        Initializes the RainfallPredictor.

        Args:
            timesteps (np.array): Array of timestamps.
            pwv_data (np.array): Array of Precipitable Water Vapor data.
            rainfall_data (np.array): Array of rainfall data.
        """
        self.df = pd.DataFrame({
            'timestamp': timesteps,
            'pwv': pwv_data,
            'rainfall': rainfall_data
        })
        self.df.set_index('timestamp', inplace=True)
        print("✅ RainfallPredictor initialized with data.")

    @staticmethod
    def simulate_data(days=10, interval_minutes=10):
        """
        Simulates a dataset of PWV and corresponding rainfall.
        """
        print("--- Simulating PWV and Rainfall Data ---")
        num_epochs = int(days * 24 * 60 / interval_minutes)
        timesteps = np.arange(num_epochs) * interval_minutes

        # Base PWV with diurnal cycle and noise
        base_pwv = 15
        diurnal_pwv = 5 * np.sin(2 * np.pi * timesteps / (24 * 60))
        noise_pwv = np.random.normal(0, 0.5, num_epochs)
        pwv = base_pwv + diurnal_pwv + noise_pwv

        # Add several sharp "pre-rain" spikes to PWV
        num_events = days * 2 # 2 rain events per day on average
        rainfall = np.zeros(num_epochs)
        for _ in range(num_events):
            event_start = np.random.randint(0, num_epochs - 120) # ensure space for event
            spike_duration = 6 # 60 minutes
            rain_delay = np.random.randint(3, 6) # 30-60 mins after spike starts
            rain_duration = 3

            # PWV spike
            spike_magnitude = np.random.uniform(5, 10)
            pwv[event_start : event_start + spike_duration] += spike_magnitude * np.sin(np.pi * np.arange(spike_duration) / spike_duration)

            # Corresponding rainfall
            rain_start = event_start + rain_delay
            rainfall[rain_start : rain_start + rain_duration] = np.random.uniform(1, 5) # mm/hr

        print(f"✅ Simulated {days} days of data with {num_events} rain events.")
        return timesteps, pwv, rainfall


        print(f"✅ Simulated {days} days of data with {num_events} rain events.")
        return timesteps, pwv, rainfall

    def engineer_features(self, windows=[1, 3, 6]):
        """
        Engineers features from the PWV data.
        - pwv_rate: 1st derivative (rate of change)
        - pwv_accel: 2nd derivative (acceleration)
        """
        print("--- Engineering Features from PWV data ---")
        self.df['pwv_rate'] = self.df['pwv'].diff()
        self.df['pwv_accel'] = self.df['pwv_rate'].diff()

        # Add features for different time windows (e.g., change over last 30 mins)
        for w in windows:
            self.df[f'pwv_rate_{w*10}m'] = self.df['pwv'].diff(periods=w)

        # Drop initial NaN values created by diff()
        self.df.dropna(inplace=True)
        print("✅ Features created: pwv_rate, pwv_accel")
        return self.df

    def label_data(self, look_ahead_minutes=30, interval_minutes=10):
        """
        Creates the target variable 'rain_in_future'.
        '1' if it rains within the look-ahead window, '0' otherwise.
        """
        print(f"--- Labeling Data (predicting rain in the next {look_ahead_minutes} mins) ---")
        self.df['rain_in_future'] = self.df['rainfall'].rolling(
            window=int(look_ahead_minutes / interval_minutes),
            min_periods=1
        ).max().shift(-int(look_ahead_minutes / interval_minutes) + 1) > 0

        # The rolling/shift operation creates NaNs at the end, remove them
        self.df.dropna(inplace=True)
        print("✅ Data labeled.")
        return self.df

    def train_and_evaluate(self, test_size=0.3, random_state=42):
        """
        Trains a logistic regression model and evaluates its performance.
        """
        print("--- Training and Evaluating Model ---")
        feature_cols = [col for col in self.df.columns if 'pwv' in col]
        X = self.df[feature_cols]
        y = self.df['rain_in_future']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        print(f"✅ Data split into {len(X_train)} training and {len(X_test)} testing samples.")

        # Train model
        model = LogisticRegression(random_state=random_state, class_weight='balanced')
        model.fit(X_train, y_train)
        print("✅ Logistic Regression model trained.")

        # Evaluate model
        y_pred = model.predict(X_test)

        print("\n--- Model Evaluation Results ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print(f"Precision: {precision_score(y_test, y_pred):.3f}")
        print(f"Recall: {recall_score(y_test, y_pred):.3f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return model


if __name__ == '__main__':
    # 1. Simulate Data
    timestamps, pwv_data, rainfall_data = RainfallPredictor.simulate_data(days=30, interval_minutes=10)

    # Initialize the predictor with the simulated data
    predictor = RainfallPredictor(timestamps, pwv_data, rainfall_data)

    # 2. Feature Engineering
    predictor.engineer_features()

    # 3. Data Labeling
    predictor.label_data(look_ahead_minutes=30)

    # 4. Model Training and Evaluation
    predictor.train_and_evaluate()

    print("\n--- Rainfall Prediction Module Complete ---")
