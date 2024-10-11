import pandas as pd
import numpy as np

from model import *

def normal(data):
    avg = np.mean(data)
    std = np.std(data)
    davg = (data - avg) / std
    return davg, avg, std

def load_data():
    # reading the data and formatting in my system
    data = pd.read_csv('data_daily.csv', parse_dates=["# Date"])
    data = data.rename(columns={"# Date": "date", "Receipt_Count": "count"})
    data = data.set_index("date")
    monthly_actuals = data.resample('M').sum()
    monthly_actuals = monthly_actuals['count']
    data = data.reset_index()
    data['day_number'] = data['date'].dt.dayofyear.astype(float)
    
    counts = data['count'].values.astype(float)
    days = data['day_number'].values.astype(float)
    #normalizing the data
    scaled, davg, dstd = normal(np.log1p(counts))
    dscaled, ddavg, ddstd = normal(days)
    features_scaled = np.column_stack((scaled, dscaled))
    return scaled, davg, dstd, dscaled, ddavg, ddstd, features_scaled, monthly_actuals, data

def load_model():
    #loading the model
    model = LSTM()
    model.load_state_dict(torch.load('LSTM_best.pth',  map_location=torch.device('cpu')),)
    model.eval()
    return model