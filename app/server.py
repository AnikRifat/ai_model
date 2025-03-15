from prophet import Prophet
from flask import Flask, request, jsonify, send_file
import pandas as pd
import os
import pickle
import logging
from datetime import datetime, timedelta

app = Flask(__name__)
model_dir = "/app/models"
os.makedirs(model_dir, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Malaysian holidays for 2023-2025 (as provided)
holidays = pd.DataFrame({
    'holiday': [
        # 2023 Holidays
        'New Year\'s Day', 'Chinese New Year', 'Chinese New Year (Second Day)', 'Federal Territory Day',
        'Prophet Muhammad\'s Birthday', 'Good Friday', 'Hari Raya Aidilfitri', 'Hari Raya Aidilfitri (Second Day)',
        'Labour Day', 'Wesak Day', 'Agong\'s Birthday', 'Hari Raya Aidiladha', 'Awal Muharram', 'National Day',
        'Malaysia Day', 'Mid-Autumn Festival', 'Deepavali', 'Maulidur Rasul', 'Christmas Day',
        # 2024 Holidays
        'New Year\'s Day', 'Chinese New Year', 'Chinese New Year (Second Day)', 'Federal Territory Day',
        'Prophet Muhammad\'s Birthday', 'Good Friday', 'Hari Raya Aidilfitri', 'Hari Raya Aidilfitri (Second Day)',
        'Labour Day', 'Wesak Day', 'Agong\'s Birthday', 'Hari Raya Aidiladha', 'Awal Muharram', 'National Day',
        'Malaysia Day', 'Mid-Autumn Festival', 'Deepavali', 'Maulidur Rasul', 'Christmas Day',
        # 2025 Holidays
        'New Year\'s Day', 'Chinese New Year', 'Chinese New Year (Second Day)', 'Federal Territory Day',
        'Shab-e-Barat', 'International Mother Language Day', 'Prophet Muhammad\'s Birthday', 'Good Friday',
        'Labour Day', 'Wesak Day', 'Agong\'s Birthday', 'Hari Raya Aidiladha', 'Awal Muharram', 'National Day',
        'Malaysia Day', 'Mid-Autumn Festival', 'Deepavali', 'Maulidur Rasul', 'Christmas Day'
    ],
    'ds': pd.to_datetime([
        # 2023 Dates
        '2023-01-01', '2023-01-24', '2023-01-25', '2023-02-01', '2023-02-22', '2023-04-07', '2023-04-22',
        '2023-04-23', '2023-05-01', '2023-05-04', '2023-06-02', '2023-06-29', '2023-07-19', '2023-08-31',
        '2023-09-16', '2023-09-28', '2023-10-29', '2023-11-14', '2023-12-25',
        # 2024 Dates
        '2024-01-01', '2024-02-10', '2024-02-11', '2024-02-01', '2024-03-18', '2024-03-29', '2024-04-10',
        '2024-04-11', '2024-05-01', '2024-05-23', '2024-06-01', '2024-06-17', '2024-07-07', '2024-08-31',
        '2024-09-16', '2024-09-17', '2024-10-26', '2024-11-09', '2024-12-25',
        # 2025 Dates
        '2025-01-01', '2025-01-29', '2025-01-30', '2025-02-01', '2025-02-14', '2025-02-21', '2025-03-21',
        '2025-04-18', '2025-05-01', '2025-05-21', '2025-06-07', '2025-06-28', '2025-07-17', '2025-08-31',
        '2025-09-16', '2025-09-17', '2025-10-23', '2025-11-08', '2025-12-25'
    ]),
    'lower_window': 0,
    'upper_window': 0,
})

# Major holidays with significant impact
major_holidays = [
    'Chinese New Year', 'Chinese New Year (Second Day)', 'Hari Raya Aidilfitri',
    'Hari Raya Aidilfitri (Second Day)', 'Hari Raya Aidiladha', 'National Day', 'Malaysia Day'
]

# Load your hotel reservation data (assuming it’s in a CSV or similar format)
# Replace this with your actual data loading method if different
data = pd.read_excel('hotel_reservations.xlsx') # Placeholder: Replace with your data source
data['Check-in date'] = pd.to_datetime(data['Check-in date'], format='%d %b %Y', errors='coerce')

# Filter for confirmed bookings only
confirmed_data = data[data['Status'] == 'Confirmed'].copy()

# Train or load Prophet models
models = {}

def train_prophet_model(room_type):
    # Prepare data for the specific room type
    df = confirmed_data[confirmed_data['Room type'] == room_type][['Check-in date', 'Sales price']].dropna()
    df.columns = ['ds', 'y']
    
    # Initialize Prophet with holidays
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        holidays=holidays
    )
    
    # Add stronger effect for major holidays
    model.add_country_holidays(country_name='MY')
    for holiday in major_holidays:
        model.holidays.loc[model.holidays['holiday'] == holiday, 'lower_window'] = -1
        model.holidays.loc[model.holidays['holiday'] == holiday, 'upper_window'] = 1
    
    # Fit the model
    model.fit(df)
    
    # Save the model
    model_path = os.path.join(model_dir, f"{room_type.replace(' ', '_')}_prophet_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model

# Train models for each room type if not already trained
room_types = confirmed_data['Room type'].unique()
for room_type in room_types:
    if not os.path.exists(os.path.join(model_dir, f"{room_type.replace(' ', '_')}_prophet_model.pkl")):
        logger.info(f"Training model for {room_type}")
        models[room_type] = train_prophet_model(room_type)
    else:
        logger.info(f"Loading model for {room_type}")
        with open(os.path.join(model_dir, f"{room_type.replace(' ', '_')}_prophet_model.pkl"), 'rb') as f:
            models[room_type] = pickle.load(f)

# API endpoint to generate and download CSV
@app.route('/generate_forecast', methods=['GET'])
def generate_forecast():
    # Forecast period: Next 30 days from March 20, 2025
    current_date = datetime(2025, 3, 20)
    future_dates = pd.date_range(start=current_date + timedelta(days=1), periods=30, freq='D')
    
    # Historical average prices
    historical_avg = confirmed_data.groupby('Room type')['Sales price'].mean().to_dict()
    
    # Generate forecasts
    forecast_data = []
    for room_type, model in models.items():
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        forecast = forecast.tail(30).reset_index(drop=True)  # Last 30 days
        
        for i, row in forecast.iterrows():
            date = row['ds']
            projected_price = max(row['yhat'], 0)  # Ensure no negative prices
            historical_price = historical_avg.get(room_type, 0)
            
            # Determine reason based on trends and holidays
            day_of_week = date.strftime('%A')
            holiday_effect = holidays[holidays['ds'] == date]['holiday'].values
            reason = f"{day_of_week} - baseline demand"
            if holiday_effect.size > 0:
                reason = f"{day_of_week} - {holiday_effect[0]} holiday impact"
            elif day_of_week in ['Friday', 'Saturday']:
                reason = f"{day_of_week} - peak weekend demand"
            
            forecast_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Room Type': room_type,
                'Projected Price (RM)': round(projected_price, 2),
                'Historical Avg Price (RM)': round(historical_price, 2),
                'Reason for Projected Price': reason
            })
    
    # Create DataFrame and save to CSV
    forecast_df = pd.DataFrame(forecast_data)
    csv_path = '/tmp/hotel_price_forecast.csv'
    forecast_df.to_csv(csv_path, index=False)
    
    # Return CSV file for download
    return send_file(
        csv_path,
        mimetype='text/csv',
        as_attachment=True,
        download_name='hotel_price_forecast.csv'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)