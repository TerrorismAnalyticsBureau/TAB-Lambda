import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime, timedelta
import calendar
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import json

# ------------------------------ Data Preprocessing ------------------------------

# Set pyTorch local env to use segmented GPU memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Clear GPU cache & Set the device to use GPU
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
# Skip rows = 1 because those are the column names
X = np.array([])

# Read the file using its encoding
data = pd.read_csv('./dataset/globalterrorismdb_0718dist.csv', encoding="Windows-1252")

# Extract relevant columns (adjust indices or column names as needed)
input_columns = data.iloc[:, [1, 2, 3, 7, 11]]
input_columns = input_columns.fillna(0)

# Convert non-numeric to numeric and fill missing values
for col in input_columns.columns:
    input_columns[col] = pd.to_numeric(input_columns[col], errors='coerce')  # Convert non-numeric to NaN
input_columns = input_columns.fillna(0)  # Replace NaN with 0

attack_target = data.iloc[:, [28]]
group_target = data.iloc[:, [58]]

# Set the base date (last day of 2017)
last_date = datetime(2017, 12, 31)

# Convert last date to numeric form
last_date_numeric = last_date.toordinal()

# Get date from dataset
data['date_str'] = data['iyear'].astype(str) + '-' + data['imonth'].astype(str).str.zfill(2) + '-' + data['iday'].astype(str).str.zfill(2)
data['date'] = pd.to_datetime(data['date_str'], errors='coerce')


# Convert dates to numeric by subtracting the last date of 2017
# Get number of days since Dec 31, 2017
data['date_numeric'] = (data['date'] - last_date).dt.days

# Extract unique values
unique_attacks = list(set(data['attacktype1_txt']))
unique_groups = list(set(data['gname']))
unique_provstates = list(set(data['provstate']))
unique_cities = list(set(data['city']))

# Initialize LabelEncoder and fit to the unique groups
attack_encoder = LabelEncoder()
attack_encoder.fit(unique_attacks)

group_encoder = LabelEncoder()
group_encoder.fit(unique_groups)

provstate_encoder = LabelEncoder()
provstate_encoder.fit(unique_provstates)

city_encoder = LabelEncoder()
city_encoder.fit(unique_cities)

# Set the output size based on the number of unique attack types
num_attack_types = len(unique_attacks)
num_groups = len(unique_groups)
num_cities = len(unique_cities)
num_provstates = len(unique_provstates)

# Create a dictionary to map names to their encoded IDs
group_dict = pd.Series(group_encoder.transform(unique_groups), index=unique_groups)
provstate_dict = pd.Series(provstate_encoder.transform(unique_provstates), index=unique_provstates)
city_dict = pd.Series(city_encoder.transform(unique_cities), index=unique_cities)

# Assign values to tensors
input_tensor = torch.tensor(input_columns.to_numpy(), dtype=torch.float32)
attack_target_tensor = torch.tensor(attack_target.values, dtype=torch.float32)
group_target_tensor = torch.tensor(group_encoder.fit_transform(group_target.values), dtype=torch.float32)
city_target_tensor = torch.tensor(city_encoder.fit_transform(data['city'].values), dtype=torch.float32)
provstate_target_tensor = torch.tensor(provstate_encoder.fit_transform(data['provstate'].values), dtype=torch.float32)

# TESTING - PRINT DICTIONARY ITEMS
#for key, value in group_dict.items():
#  print("group: ", key, "| ID #:", value)

#for key, value in provstate_dict.items():
#  print("provstate: ", key, "| ID #:", value)

#for key, value in city_dict.items():
#  print("city: ", key, "| ID #:", value)

# Assign values to tensors for processing
X_tensor = input_tensor

# Normalize: mean and std for each feature
mean = X_tensor.mean(dim=0, keepdim=True)
std = X_tensor.std(dim=0, keepdim=True)
X_tensor = (X_tensor - mean) / std

Y_tensor_attack = attack_target_tensor
Y_tensor_group = group_target_tensor
Y_tensor_city = city_target_tensor
Y_tensor_provstate = provstate_target_tensor
Y_tensor_date = torch.tensor(data['date_numeric'] - last_date_numeric, dtype=torch.float32)

# Set tensors to use GPU
X_tensor = X_tensor.to(device)
Y_tensor_attack = Y_tensor_attack.to(device)
Y_tensor_group = Y_tensor_group.to(device)
Y_tensor_city = Y_tensor_city.to(device)
Y_tensor_provstate = Y_tensor_provstate.to(device)
Y_tensor_date = Y_tensor_date.to(device)

# ------------------------------ LSTM Prediction Model ------------------------------

def train_model(X_tensor, Y_tensor, num_classes, sequence_length=30, hidden_size=128, num_epochs=10, batch_size=32):
    class LSTMPredictor(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMPredictor, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            lstm_out = self.dropout(lstm_out)
            logits = self.fc(lstm_out[:, -1, :])
            return logits

    # Create sequences
    def create_sequences(input_data, seq_length):
        sequences = []
        for i in range(len(input_data) - seq_length + 1):
            seq = input_data[i:i + seq_length]
            sequences.append(seq)
        return torch.stack(sequences)

    sequences = create_sequences(X_tensor, sequence_length)

    # Create DataLoader
    dataset = TensorDataset(sequences, Y_tensor[:len(sequences)])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = LSTMPredictor(input_size=X_tensor.shape[1], hidden_size=hidden_size, output_size=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Training loop
    for epoch in range(num_epochs):
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_x)
            if batch_y.ndim > 1:
                batch_y = batch_y.argmax(dim=1)
            batch_y = batch_y.long()

            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model

# ------------------------------ Date Prediction Model ------------------------------

class LSTMDate(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMDate, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# Convert date to numeric since the final day in dataset
def convert_date_to_numeric(date):
    return (date - datetime(2017, 12, 31)).days

# Generate date range (years, months, days)
def generate_date_range(start_year, end_year):
    date_list = []
    for year in range(start_year, end_year + 1):
      # Loop through months 1 to 12
        for month in range(1, 13):
            num_days = calendar.monthrange(year, month)[1]
            # Loop through the days of the month
            for day in range(1, num_days + 1):
                date = datetime(year, month, day)
                date_list.append(date)
    return date_list

# Create sequences from the numeric dates and features
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Main function to train the model
def train_date(X_tensor, Y_tensor, sequence_length=30, hidden_size=128, num_epochs=1000, batch_size=32):
    X_tensor = X_tensor.to(device)
    Y_tensor = Y_tensor.to(device)

    # Reshape X_tensor to have shape [num_samples, sequence_length, num_features]
    X_tensor = X_tensor.reshape(X_tensor.shape[0], X_tensor.shape[1], 1)

    # Create a DataLoader for batching
    dataset = TensorDataset(X_tensor, Y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create the model
    input_size = X_tensor.shape[2]  # Number of features (year, month, day)
    output_size = 1  # Predicting a single value (days since reference date)
    model = LSTMDate(input_size, hidden_size, output_size)

    # Move model to device
    model.to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        for i, (inputs, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Zero the gradients, backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (i + 1) % 100 == 0:  # Print every 100 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}], Loss: {loss.item():.4f}")

    return model

# Generate date list for the years 2018 to 2023
date_list = generate_date_range(2018, 2023)

# Convert the list of dates to numeric values (days since 2017-12-31)
date_numeric = [convert_date_to_numeric(date) for date in date_list]

# Create sequences for training
# Use the last 30 days to predict the next one
sequence_length = 30
X, Y = create_sequences(date_numeric, sequence_length)

# Convert to PyTorch tensors
X_tensor_date = torch.tensor(X, dtype=torch.float32)
Y_tensor_date = torch.tensor(Y, dtype=torch.float32)

# ------------------------------ Train & Evaluate Models ------------------------------
print("Training attack prediction")
model_attack = train_model(X_tensor, Y_tensor_attack, num_classes=num_attack_types)
model_attack = model_attack.to(device)

print("Training group prediction")
model_groups = train_model(X_tensor, Y_tensor_group, num_classes=num_groups)
model_groups = model_groups.to(device)

print("Training city prediction")
model_city = train_model(X_tensor, Y_tensor_city, num_classes=num_cities)
model_city = model_city.to(device)

print("Training province/ state prediction")
model_provstate = train_model(X_tensor, Y_tensor_provstate, num_classes=num_provstates)
model_provstate = model_provstate.to(device)

print("Training date prediction")
model_date = train_date(X_tensor_date, Y_tensor_date, sequence_length=sequence_length)
model_date = model_date.to(device)

# Set the models to evaluation mode
model_attack.eval()
model_groups.eval()
model_city.eval()
model_provstate.eval()
model_date.eval()
print("Training complete")
# ------------------------------ Testing ------------------------------
with torch.no_grad():
    # Prepare the most recent sequence for prediction
    recent_sequence = X_tensor[-1:].unsqueeze(0)  # Add batch dimension

    # Model 1: Attack prediction
    prediction_attack = model_attack(recent_sequence)  # Get model's prediction (logits)
     # Get the predicted class (argmax of logits)
    predicted_class_attack = torch.argmax(prediction_attack, dim=1).item()  # Convert logits to class index
    # Decode the predicted class back to attack type using the encoder
    attack_type = attack_encoder.inverse_transform([predicted_class_attack])
    print("Predicted Attack Type:", attack_type[0])

    # Model 2: Group prediction
    prediction_group = model_groups(recent_sequence)  # Get model's prediction (logits)
    # Get the predicted class (argmax of logits)
    predicted_class_group = torch.argmax(prediction_group, dim=1).item()  # Convert logits to class index
    # Decode the predicted class back to attack type using the encoder
    group_name = group_encoder.inverse_transform([predicted_class_group])
    print("Predicted Group Name:", group_name[0])

    # Model 3: City prediction
    prediction_city = model_city(recent_sequence)  # Get model's prediction (logits)
    # Get the predicted class (argmax of logits)
    predicted_class_city = torch.argmax(prediction_city, dim=1).item()  # Convert logits to class index
    # Decode the predicted class back to attack type using the encoder
    city_name = city_encoder.inverse_transform([predicted_class_city])
    print("Predicted City Name:", city_name[0])

    # Model 4: provstate prediction
    prediction_provstate = model_provstate(recent_sequence)  # Get model's prediction (logits)
    # Get the predicted class (argmax of logits)
    predicted_class_provstate = torch.argmax(prediction_provstate, dim=1).item()  # Convert logits to class index
    # Decode the predicted class back to attack type using the encoder
    provstate_name = provstate_encoder.inverse_transform([predicted_class_provstate])
    print("Predicted State Name:", provstate_name[0])

    # Model 5: Date prediction
    # Get 'raw' values from recent sequence
    recent_sequence = torch.nan_to_num(recent_sequence, nan=0.0)
    recent_sequence = recent_sequence[0][0]

    # Convert to PyTorch tensor and reshape
    recent_sequence = torch.tensor(recent_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

    # Predict the offset (number of days)
    predicted_offset = model_date(recent_sequence).item()
    # Convert the offset to a predicted date
    predicted_date = datetime(2017, 12, 31) + timedelta(days=predicted_offset)

    # Print the predicted date
    print(f"Predicted Date: {predicted_date}")

# Saved data
saved_predictions = {
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "attack_type": attack_type[0],
    "group_name": group_name[0],
    "city_name": city_name[0],
    "provstate_name": provstate_name[0],
    "predicted_date": predicted_date.strftime('%Y-%m-%d')  # Convert datetime to string
}

# Save to a JSON file
with open('predictions.json', 'w') as f:
    json.dump(saved_predictions, f, indent=4)

print("Saved predictions to predictions.json")