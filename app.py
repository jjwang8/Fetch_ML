from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime

from model import *
from util import *

app = Flask(__name__)

sequence_length = 60  
model = None
data = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to plot predictions vs real values
def plot_predictions_vs_real(real, predicted):
    real.set_index('date', inplace=True)
    dates_2022 = pd.date_range(start='2022-01-01', periods=len(predicted))
    predicted_data = pd.DataFrame({'date': dates_2022, 'count': predicted})

    predicted_data.set_index('date', inplace=True)
    plt.figure(figsize=(12, 6))
    plt.plot(real.index, real['count'], label='Actual Receipts (2021)')
    plt.plot(predicted_data.index, predicted_data['count'], label='Predicted Data')
    plt.title('Predictions vs Actual Receipts (Daily)')
    plt.xlabel('Time')
    plt.ylabel('Number of Receipts')
    plt.legend()
    plt.savefig('static/prediction_vs_real.png')
    plt.close()

# Function to plot monthly bar chart
def plot_monthly_bar_chart(monthly_predictions):
    global monthly_actuals

    # Combine actuals and predictions
    monthly_predictions = monthly_predictions.reset_index()
    combined_data = pd.concat([monthly_actuals.reset_index(), monthly_predictions], axis=0, ignore_index=True)
    combined_data.columns = ['date', 'receipts']
    
    combined_data['Type'] = combined_data['date'].apply(lambda x: 'Actual' if x.year == 2021 else 'Predicted')
    # Plot
    plt.figure(figsize=(12, 6))
    # Separate actual and predicted data
    actual_data = combined_data[combined_data['Type'] == 'Actual']
    predicted_data = combined_data[combined_data['Type'] == 'Predicted']
    # Plot bars
    plt.bar(actual_data['date'].dt.strftime('%Y-%m'), actual_data['receipts'], label='Actual Receipts', color='blue')
    plt.bar(predicted_data['date'].dt.strftime('%Y-%m'), predicted_data['receipts'], label='Predicted Receipts', color='orange')
    plt.title('Monthly Scanned Receipts: Actual vs Predicted')
    plt.xlabel('Month')
    plt.ylabel('Total Scanned Receipts')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/monthly_bar_chart.png')
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the date input from the user
        end_date_str = request.form['end_date']
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            if end_date.year <= 2021:
                error_message = 'Please enter a date after 2021-12-31.'
                return render_template('index.html', error_message=error_message)
            else:
                # Perform prediction up to the specified date
                predictions, monthly_predictions = predict_future(end_date)
                table_data = prepare_table_data(monthly_predictions)
                #plotting graphs for display
                plot_monthly_bar_chart(monthly_predictions)
                plot_predictions_vs_real(data.copy(), predictions)
                return render_template('index.html', 
                                       predictions_available=True, 
                                       end_date=end_date_str,
                                       table_data=table_data)
        except ValueError as e:
            error_message = 'Invalid date format. Please use YYYY-MM-DD.'
            return render_template('index.html', error_message=error_message)
    else:
        return render_template('index.html')
    
def prepare_table_data(monthly_predictions):
    monthly_predictions = monthly_predictions.reset_index()
    monthly_predictions['Month'] = monthly_predictions['date'].dt.strftime('%Y-%m')
    table_data = monthly_predictions.to_dict(orient='records')
    return table_data

def predict_future(end_date):
    global model, data_min, data_max, features_scale
    model.eval()
    # getting last 60 data points to start
    last_sequence = features_scale[-sequence_length:] 

    day_numbers_2022 = list(range(366, 366+(end_date- datetime(2022, 1, 1)).days ))

    # normal the day number
    day_numbers_2022_scaled = (day_numbers_2022 - ddavg) / ddstd
    predictions = []
    for i in range(len(day_numbers_2022_scaled)):
        day_num = day_numbers_2022_scaled[i]
        input_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_seq)
            predicted_receipt = output.item()
            predictions.append(predicted_receipt)
        new_entry = np.array([predicted_receipt, day_num])
        last_sequence = np.vstack((last_sequence[1:], new_entry))
    

    # reverse predictions
    predictions_inverse = np.expm1(np.array(predictions) * dstd + davg)
    # Make some dates
    dates_future = pd.date_range(start='2022-01-01', periods=len(day_numbers_2022))
    predicted_data = pd.DataFrame({'date': dates_future, 'count': predictions_inverse})
    # monthly sums
    predicted_data.set_index('date', inplace=True)
    monthly_predictions = predicted_data.resample('M').sum()
    return predictions_inverse, monthly_predictions

if __name__ == '__main__':
    # Load data and model before starting the app
    scaled, davg, dstd, dscale, ddavg, ddstd, features_scale, monthly_actuals, data = load_data()
    model = load_model().to(device)
    app.run(debug=True)