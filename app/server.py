from prophet import Prophet
from flask import Flask, request, jsonify, send_file
import pandas as pd
import os
import pickle
import logging
from datetime import datetime, timedelta
import random

# Set a fixed seed for reproducibility
random.seed(42)  # Ensures consistent random values across runs

app = Flask(__name__)
model_dir = "/app/models"
os.makedirs(model_dir, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Malaysian holidays for 2023-2025
holidays = pd.DataFrame({
    'holiday': [
        'New Year\'s Day', 'Chinese New Year', 'Chinese New Year (Second Day)', 'Federal Territory Day',
        'Prophet Muhammad\'s Birthday', 'Good Friday', 'Hari Raya Aidilfitri', 'Hari Raya Aidilfitri (Second Day)',
        'Labour Day', 'Wesak Day', 'Agong\'s Birthday', 'Hari Raya Aidiladha', 'Awal Muharram', 'National Day',
        'Malaysia Day', 'Mid-Autumn Festival', 'Deepavali', 'Maulidur Rasul', 'Christmas Day',
        'New Year\'s Day', 'Chinese New Year', 'Chinese New Year (Second Day)', 'Federal Territory Day',
        'Prophet Muhammad\'s Birthday', 'Good Friday', 'Hari Raya Aidilfitri', 'Hari Raya Aidilfitri (Second Day)',
        'Labour Day', 'Wesak Day', 'Agong\'s Birthday', 'Hari Raya Aidiladha', 'Awal Muharram', 'National Day',
        'Malaysia Day', 'Mid-Autumn Festival', 'Deepavali', 'Maulidur Rasul', 'Christmas Day',
        'New Year\'s Day', 'Chinese New Year', 'Chinese New Year (Second Day)', 'Federal Territory Day',
        'Shab-e-Barat', 'International Mother Language Day', 'Prophet Muhammad\'s Birthday', 'Good Friday',
        'Labour Day', 'Wesak Day', 'Agong\'s Birthday', 'Hari Raya Aidiladha', 'Awal Muharram', 'National Day',
        'Malaysia Day', 'Mid-Autumn Festival', 'Deepavali', 'Maulidur Rasul', 'Christmas Day'
    ],
    'ds': pd.to_datetime([
        '2023-01-01', '2023-01-24', '2023-01-25', '2023-02-01', '2023-02-22', '2023-04-07', '2023-04-22',
        '2023-04-23', '2023-05-01', '2023-05-04', '2023-06-02', '2023-06-29', '2023-07-19', '2023-08-31',
        '2023-09-16', '2023-09-28', '2023-10-29', '2023-11-14', '2023-12-25',
        '2024-01-01', '2024-02-10', '2024-02-11', '2024-02-01', '2024-03-18', '2024-03-29', '2024-04-10',
        '2024-04-11', '2024-05-01', '2024-05-23', '2024-06-01', '2024-06-17', '2024-07-07', '2024-08-31',
        '2024-09-16', '2024-09-17', '2024-10-26', '2024-11-09', '2024-12-25',
        '2025-01-01', '2025-01-29', '2025-01-30', '2025-02-01', '2025-02-14', '2025-02-21', '2025-03-21',
        '2025-04-18', '2025-05-01', '2025-05-21', '2025-06-07', '2025-06-28', '2025-07-17', '2025-08-31',
        '2025-09-16', '2025-09-17', '2025-10-23', '2025-11-08', '2025-12-25'
    ]),
    'lower_window': 0,
    'upper_window': 0,
})

major_holidays = [
    'Chinese New Year', 'Chinese New Year (Second Day)', 'Hari Raya Aidilfitri',
    'Hari Raya Aidilfitri (Second Day)', 'Hari Raya Aidiladha', 'National Day', 'Malaysia Day'
]

# Load hotel reservation data from Excel file within the container
data_path = "/app/hotel_reservations.xlsx"
data = pd.read_excel(data_path)
data['Check-in date'] = pd.to_datetime(data['Check-in date'], format='%d %b %Y', errors='coerce')

# Filter for confirmed bookings only
confirmed_data = data[data['Status'] == 'Confirmed'].copy()

# Simulated admin settings (replace with actual admin panel data in production)
floor_prices = {'Deluxe No Window': 70.00, 'Superior No Window': 60.00, 'Family 3 No Window kiosk': 100.00}
ceiling_prices = {'Deluxe No Window': 150.00, 'Superior No Window': 120.00, 'Family 3 No Window kiosk': 180.00}
base_rates = floor_prices  # Assuming base rate is the floor rate
historical_avg = confirmed_data.groupby('Room type')['Sales price'].mean().to_dict()

# Simulated competitor rates (replace with RAPIDAPI integration in production)
def get_competitor_rate(room_type, date):
    base = historical_avg.get(room_type, 100)
    return base * (1 + random.uniform(-0.2, 0.2))  # ±20% variation, consistent with seed

# Train or load Prophet models
models = {}

def train_prophet_model(room_type):
    df = confirmed_data[confirmed_data['Room type'] == room_type][['Check-in date', 'Sales price']].dropna()
    df.columns = ['ds', 'y']
    
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        holidays=holidays
    )
    
    model.add_country_holidays(country_name='MY')
    for holiday in major_holidays:
        model.holidays.loc[model.holidays['holiday'] == holiday, 'lower_window'] = -1
        model.holidays.loc[model.holidays['holiday'] == holiday, 'upper_window'] = 1
    
    try:
        model.fit(df)
        model_path = os.path.join(model_dir, f"{room_type.replace(' ', '_')}_prophet_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return model
    except Exception as e:
        logger.error(f"Error training model for {room_type}: {str(e)}")
        return None

# Train or load models
room_types = confirmed_data['Room type'].unique()
for room_type in room_types:
    model_path = os.path.join(model_dir, f"{room_type.replace(' ', '_')}_prophet_model.pkl")
    if not os.path.exists(model_path):
        logger.info(f"Training model for {room_type}")
        model = train_prophet_model(room_type)
        if model:
            models[room_type] = model
    else:
        logger.info(f"Loading model for {room_type}")
        with open(model_path, 'rb') as f:
            models[room_type] = pickle.load(f)

# Dynamic pricing logic based on holidays and competitor rates
def calculate_projected_price(room_type, date, prophet_price, competitor_rate):
    floor = floor_prices.get(room_type, 50.00)
    ceiling = ceiling_prices.get(room_type, 200.00)
    base = base_rates.get(room_type, floor)
    
    # Check for holiday effect
    holiday_effect = holidays[holidays['ds'] == date]['holiday'].values
    holiday_str = f" due to {holiday_effect[0]}" if holiday_effect.size > 0 else ""
    
    # Base price on competitor rate
    price = competitor_rate
    reason = f"Set to match nearby hotels{holiday_str}"
    
    # Adjust price for holidays
    if holiday_effect.size > 0:
        price *= 1.10  # 10% increase for holidays
        reason += "; raised slightly for holiday demand"
    
    # Ensure price stays within bounds
    price = min(max(price, floor), ceiling)
    return round(price, 2), reason

@app.route('/generate_forecast', methods=['GET'])
def generate_forecast():
    current_date = datetime(2025, 3, 15)  # Hardcoded for consistency with your examples
    future_dates = pd.date_range(start=current_date + timedelta(days=1), periods=30, freq='D')
    
    forecast_data = []
    for room_type, model in models.items():
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        forecast = forecast.tail(30).reset_index(drop=True)
        
        for i, row in forecast.iterrows():
            date = row['ds']
            prophet_price = max(row['yhat'], 0)
            competitor_rate = get_competitor_rate(room_type, date)
            projected_price, reason = calculate_projected_price(room_type, date, prophet_price, competitor_rate)
            
            forecast_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Room Type': room_type,
                'Projected Price (RM)': projected_price,
                'Historical Avg Price (RM)': round(historical_avg.get(room_type, prophet_price), 2),
                'Reason for Projected Price': reason
            })
    
    forecast_df = pd.DataFrame(forecast_data)
    csv_path = '/tmp/hotel_price_forecast.csv'
    forecast_df.to_csv(csv_path, index=False)
    
    return send_file(
        csv_path,
        mimetype='text/csv',
        as_attachment=True,
        download_name='hotel_price_forecast.csv'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)