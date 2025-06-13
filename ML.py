### **1️⃣ Load Required Libraries & Dataset**

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import requests  # For fetching real-time climate data
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("climate_change_indicators.csv")

# Display first few rows to understand data structure
print(df.head())

### **2️⃣ Data Cleaning & Handling Missing Values**

print(df.dtypes)

# Convert relevant columns to numeric (excluding text-based columns)
df.iloc[:, 9:] = df.iloc[:, 9:].apply(pd.to_numeric, errors='coerce')

# Fill missing values with the column's mean (only for numeric columns)
df.fillna(df.iloc[:, 9:].mean(numeric_only=True), inplace=True)

print(df.isnull().sum())  # Should show zero missing values

### **3️⃣ Feature Engineering**

# Select only temperature-based columns (F1961 - F2022)
numeric_columns = df.loc[:, 'F1961':'F2022']

# Compute rolling average temperature change over years
df['Rolling_Avg_Temp'] = numeric_columns.mean(axis=1)

# Compute temperature deviation from historical mean
df['Temp_Deviation'] = df['Rolling_Avg_Temp'] - df['Rolling_Avg_Temp'].mean()

### **4️⃣ Prepare Data for LSTM**

# Extract only numerical temperature change data
X = df.iloc[:, 9:].select_dtypes(include=['float64', 'float32', 'int64', 'int32']).values
y = X[:, -1]  # Predict last known temperature change

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Reshape input for LSTM (samples, time steps, features)
X_tensor = X_tensor.unsqueeze(2)

# Split dataset into training and testing sets (80% train, 20% test)
train_size = int(0.8 * len(X_tensor))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Create DataLoader for efficient training
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

### **5️⃣ Define the LSTM Model**

class ClimateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(ClimateLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output Layer

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]  # Take last time step
        return self.fc(last_step)

# Initialize LSTM model
model = ClimateLSTM(input_size=1, hidden_size=64, num_layers=2)

### **6️⃣ Set Loss Function & Optimizer**

criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

### **7️⃣ Train the LSTM Model**

epochs = 100
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

### **8️⃣ Evaluate Model Performance**

from torchmetrics.regression import R2Score

model.eval()
with torch.no_grad():
    test_predictions = model(X_test)

# Ensure shapes are correct for torchmetrics
r2_metric = R2Score()
r2 = r2_metric(test_predictions, y_test)
accuracy_percentage = r2.item() * 100
print(f"Model Accuracy (R² Score): {accuracy_percentage:.2f}%")

def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-8  # Avoid division errors
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100

mape = mean_absolute_percentage_error(y_test, test_predictions)
print(f"Mean Absolute Percentage Error (MAPE): {mape.item():.2f}%")

# Define number of future steps to forecast
future_steps = 10  # Predict the next 10 years

# Select last samples from test dataset to use for forecasting
future_input = X_test[-future_steps:].clone()

# Generate future predictions using trained model
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    future_predictions = model(future_input)

# Display forecasted temperature changes
print(f"Future Temperature Predictions: {future_predictions.numpy().flatten()}")

### **🔟 Real-Time Climate Data Integration**

url = "https://api.open-meteo.com/v1/forecast?latitude=-1.29&longitude=36.82&current=temperature_2m,wind_speed_10m"
response = requests.get(url)
data = response.json()

real_time_temp = data["current"]["temperature_2m"]
real_time_wind = data["current"]["wind_speed_10m"]

numerical_cols = df.iloc[:, 9:].select_dtypes(include=np.number)
mean_historical_temp = numerical_cols.mean().mean()

# Create a new row with the same columns as df
new_row = pd.DataFrame([{col: np.nan for col in df.columns}])
new_row['Rolling_Avg_Temp'] = real_time_temp
new_row['Temp_Deviation'] = real_time_temp - mean_historical_temp

df = pd.concat([df, new_row], ignore_index=True)
# Display the last few rows to show the appended data
print(df.tail())