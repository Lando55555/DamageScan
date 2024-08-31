# src/e3_calculations.py

import torch
import pandas as pd
from train_model import DryingTimeModel

def load_model(model_path="models/drying_time_model.pth"):
    model = DryingTimeModel()
    model.load_state_dict(torch.load(model_path))
    model.to("cuda")
    model.eval()
    return model

def predict_drying_time(features, model):
    with torch.no_grad():
        features = torch.tensor(features).float().to("cuda")
        prediction = model(features).cpu().numpy()
        return prediction[0]

def calculate_e3_dry_times(csv_data, model, weather_info):
    dry_times = {}
    total_time = 0

    for _, row in csv_data.iterrows():
        features = [
            row['ambient_temp'],
            row['dew_point_temp'],
            row['wet_bulb_temp'],
            row['gpp_ambient'],
            row['gpp_dew_point'],
            row['gpp_wet_bulb'],
            weather_info['temperature'],
            weather_info['humidity']
        ]
        predicted_time = predict_drying_time(features, model)
        dry_times[row['Room_Name']] = float(predicted_time)
        total_time += predicted_time

    dry_times['Total'] = float(total_time)
    return dry_times