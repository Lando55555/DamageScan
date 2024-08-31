# src/train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class RoomDataset(Dataset):
    def __init__(self, csv_data):
        self.data = csv_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = torch.tensor([
            row['ambient_temp'],
            row['dew_point_temp'],
            row['wet_bulb_temp'],
            row['gpp_ambient'],
            row['gpp_dew_point'],
            row['gpp_wet_bulb'],
            row['outdoor_temp'],
            row['outdoor_humidity']
        ]).float()
        target = torch.tensor(row['drying_time']).float()
        return features, target

class DryingTimeModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=20, output_size=1):
        super(DryingTimeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(csv_path, model_save_path="models/drying_time_model.pth"):
    csv_data = pd.read_csv(csv_path)
    dataset = RoomDataset(csv_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = DryingTimeModel().to("cuda")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(100):
        for features, target in dataloader:
            features, target = features.to("cuda"), target.to("cuda")
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model('data/sample_report.csv')