import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from util import *
from model import *

sequence_length = 60
batch_size = 256
lr = 0.001
num_epochs = 300
device = "cuda" if torch.cuda.is_available() else "cpu"

scaled, davg, dstd, dscaled, ddavg, ddstd, features_scaled, monthly_actuals, data = load_data()

# combining the features
def create_sequences(features, targets, seq_length):
    xs = []
    ys = []
    for i in range(len(features) - seq_length):
        x = features[i:i+seq_length]
        y = targets[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(features_scaled, scaled, sequence_length)
temp = [(X[i], y[i]) for i in range(len(X))]
import random
random.shuffle(temp)
X, y = [i[0] for i in temp], [i[1] for i in temp]

#Splitting them into train and test splits
train_size = int(len(X) * 0.8)  

X_train = X[:train_size]
y_train = y[:train_size]
X_valid = X[train_size:]
y_valid = y[train_size:]

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
valid_dataset = TimeSeriesDataset(X_valid, y_valid)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

#Setting up model and optimizer
model = LSTM()
model.to(device)
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
best_valid_loss = float('inf') 

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for sequences, targets in train_loader:
        optimizer.zero_grad()
        sequences = sequences.to("cuda")
        targets = targets.to("cuda")
        outputs = model(sequences)
        loss = criterion(outputs, targets.unsqueeze(-1))
        loss.backward()
        #Clip grads for better stability during training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())
    # Validation
    model.eval()
    valid_losses = []
    with torch.no_grad():
        for sequences, targets in valid_loader:
            sequences = sequences.to("cuda")
            targets = targets.to("cuda")
            outputs = model(sequences)
            loss = criterion(outputs, targets.unsqueeze(-1))
            valid_losses.append(loss.item())
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Training Loss: {np.mean(train_losses):.6f}, '
          f'Validation Loss: {np.mean(valid_losses):.6f}')
    #Determine the best model on validation and save it
    average_valid_loss = np.mean(valid_losses)
    if average_valid_loss < best_valid_loss:
        best_valid_loss = average_valid_loss
        # Save the best model
        torch.save(model.state_dict(), 'LSTM_best.pth')
        print(f'Validation loss decreased. Saving model at epoch {epoch+1}')