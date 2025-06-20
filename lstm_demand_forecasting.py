import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

class LSTMDemandForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMDemandForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = lstm_out[:, -1, :]
        output = self.dropout(lstm_out)
        output = self.linear(output)
        return output
class DemandForecastingSystem:
    def __init__(self, sequence_length=30, features=None):
        self.sequence_length = sequence_length
        self.features = features or [
            'price', 'competitor_price', 'seasonality', 'day_of_week',
            'promotional_activity', 'inventory_level', 'weather_index'
        ]
        self.scalers = {}
        self.model = None
    def create_sequences(self, data, target_column='demand'):
        sequences = []
        targets = []
        for i in range(len(data) - self.sequence_length):
            seq = data[self.features].iloc[i:i+self.sequence_length].values
            sequences.append(seq)
            target = data[target_column].iloc[i+self.sequence_length]
            targets.append(target)
        return np.array(sequences), np.array(targets)
    def prepare_data(self, data):
        scaled_data = data.copy()
        for feature in self.features + ['demand']:
            scaler = MinMaxScaler()
            scaled_data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1)).flatten()
            self.scalers[feature] = scaler
        return scaled_data
    def train(self, data, epochs=200, batch_size=32, learning_rate=0.001, validation_split=0.2):
        scaled_data = self.prepare_data(data)
        X, y = self.create_sequences(scaled_data)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val).unsqueeze(1)
        input_size = len(self.features)
        self.model = LSTMDemandForecaster(input_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            train_pred = self.model(X_train)
            train_loss = criterion(train_pred, y_train)
            train_loss.backward()
            optimizer.step()
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val)
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            scheduler.step(val_loss)
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}')
        return train_losses, val_losses
    def predict(self, data, steps_ahead=7):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        self.model.eval()
        scaled_data = data.copy()
        for feature in self.features:
            if feature in self.scalers:
                scaled_data[feature] = self.scalers[feature].transform(
                    data[feature].values.reshape(-1, 1)
                ).flatten()
        predictions = []
        last_sequence = scaled_data[self.features].iloc[-self.sequence_length:].values
        with torch.no_grad():
            for _ in range(steps_ahead):
                input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0)
                pred_scaled = self.model(input_tensor).item()
                pred_actual = self.scalers['demand'].inverse_transform([[pred_scaled]])[0][0]
                predictions.append(pred_actual)
                new_row = last_sequence[-1].copy()
                new_row[0] = pred_scaled
                last_sequence = np.vstack([last_sequence[1:], new_row])
        return predictions
    def evaluate(self, data):
        scaled_data = self.prepare_data(data)
        X, y_true = self.create_sequences(scaled_data)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            y_pred_scaled = self.model(X_tensor).numpy().flatten()
            y_pred = self.scalers['demand'].inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).flatten()
            y_true_actual = self.scalers['demand'].inverse_transform(
                y_true.reshape(-1, 1)
            ).flatten()
        mae = mean_absolute_error(y_true_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true_actual, y_pred))
        mape = np.mean(np.abs((y_true_actual - y_pred) / y_true_actual)) * 100
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'predictions': y_pred,
            'actual': y_true_actual
        }
def generate_sample_demand_data(days=365):
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'price': np.random.normal(50, 10, days),
        'competitor_price': np.random.normal(52, 8, days),
        'seasonality': np.sin(2 * np.pi * np.arange(days) / 365),
        'day_of_week': [d.weekday() for d in dates],
        'promotional_activity': np.random.binomial(1, 0.1, days),
        'inventory_level': np.random.normal(1000, 200, days),
        'weather_index': np.random.normal(0, 1, days)
    })
    base_demand = 1000
    price_effect = -10 * (data['price'] - 50)
    competitor_effect = 5 * (data['competitor_price'] - 52)
    seasonal_effect = 200 * data['seasonality']
    weekend_effect = 100 * (data['day_of_week'] >= 5)
    promo_effect = 300 * data['promotional_activity']
    inventory_effect = 0.1 * (data['inventory_level'] - 1000)
    weather_effect = 50 * data['weather_index']
    data['demand'] = (base_demand + price_effect + competitor_effect + seasonal_effect + weekend_effect + promo_effect + inventory_effect + weather_effect).astype(int)
    return data
