from prophet import Prophet
from flask import Flask, request, jsonify
import pandas as pd
import os
import pickle
import logging
from datetime import datetime

app = Flask(__name__)
model_dir = "/app/models"
os.makedirs(model_dir, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Room types with price ranges (RM)
room_types = {
    'Super Deluxe Twin': {'floor': 600, 'ceiling': 900, 'base': 640},
    'Super Deluxe King': {'floor': 700, 'ceiling': 1000, 'base': 700},
    'Infinity Sea View': {'floor': 900, 'ceiling': 1200, 'base': 900},
    'Junior Suite': {'floor': 1200, 'ceiling': 1600, 'base': 1200},
    'Panorama Ocean Suite': {'floor': 1600, 'ceiling': 2000, 'base': 1600},
}

# Updated Malaysian holidays for 2023-2025
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

# Train or load Prophet models
models = {}
for room_type in room_types:
    model_path = f"{model_dir}/{room_type.replace(' ', '_')}_model.pkl"
    if not os.path.exists(model_path):
        bookings = pd.read_csv("/data/bookings.csv")
        df = bookings[bookings['room_type'] == room_type][['check_in_date', 'price_per_day']].rename(
            columns={'check_in_date': 'ds', 'price_per_day': 'y'}
        )
        df['ds'] = pd.to_datetime(df['ds'])

        comp_prices = pd.read_csv("/data/competitors_room_prices.csv")
        comp_prices = comp_prices[['check_date', 'price']].rename(columns={'check_date': 'ds', 'price': 'competitor_price'})
        comp_prices['ds'] = pd.to_datetime(comp_prices['ds'])
        comp_prices = comp_prices.groupby('ds', as_index=False).agg({'competitor_price': 'mean'})

        df = pd.merge(df, comp_prices, on='ds', how='left').fillna({'competitor_price': comp_prices['competitor_price'].mean()})

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=holidays,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        model.add_regressor('competitor_price')
        model.fit(df)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    models[room_type] = model

# Realistic occupancy simulation
def get_occupancy_data(date):
    date_obj = pd.to_datetime(date)
    is_weekend = date_obj.weekday() >= 5
    is_holiday = date_obj in holidays['ds'].values
    holiday_name = holidays[holidays['ds'] == date_obj]['holiday'].iloc[0] if is_holiday else None
    is_major_holiday = holiday_name in major_holidays if holiday_name else False

    base_occupancy = 50  # Weekday base
    if is_weekend:
        base_occupancy += 20  # Weekend boost
    if is_holiday:
        base_occupancy += 30 if is_major_holiday else 10  # Major vs minor holiday boost

    occupancy = min(95, base_occupancy)  # Cap at 95%
    return {'occupancy': occupancy, 'reservations': int(228 * occupancy / 100)}

# Historical price calculation
def calculate_historical_price(occupancy, room_type, historical_data):
    base = room_types[room_type]['base']
    if historical_data:
        avg_price = sum(historical_data) / len(historical_data)
        return avg_price + (occupancy - 50) * 2 if occupancy > 50 else avg_price
    return base

# Future pricing logic
def future_price(room_type, occupancy, competitor_rate, historical_data, date_obj):
    floor_price = room_types[room_type]['floor']
    ceiling_price = room_types[room_type]['ceiling']
    reason = ""

    is_holiday = date_obj in holidays['ds'].values
    holiday_name = holidays[holidays['ds'] == date_obj]['holiday'].iloc[0] if is_holiday else None
    is_major_holiday = holiday_name in major_holidays if holiday_name else False
    is_weekend = date_obj.weekday() >= 5

    if occupancy < 50:
        price = competitor_rate if competitor_rate > floor_price else floor_price
        reason = "Matched competitor rate" if competitor_rate > floor_price else "Set to floor price"
    else:
        price = calculate_historical_price(occupancy, room_type, historical_data)
        reason = "Based on historical pricing"

    if is_holiday:
        multiplier = 1.3 if is_major_holiday else 1.1
        price *= multiplier
        reason = f"Higher due to {holiday_name}"
    elif is_weekend:
        price *= 1.1
        reason = "Higher due to weekend demand"

    price = max(floor_price, min(ceiling_price, price))
    return {'price': price, 'reason': reason}

@app.route('/predict', methods=['GET'])
def predict():
    date = request.args.get('date')
    room_type = request.args.get('room_type')
    if room_type not in room_types:
        return jsonify({'error': 'Invalid room type'}), 400

    date_obj = pd.to_datetime(date)
    booking_data = get_occupancy_data(date)
    occupancy = booking_data['occupancy']

    # Fetch competitor rate
    comp_prices = pd.read_csv("/data/competitors_room_prices.csv")
    comp_prices['check_date'] = pd.to_datetime(comp_prices['check_date'])
    comp_rates = comp_prices[comp_prices['check_date'] == date_obj]['price']
    if not comp_rates.empty:
        competitor_rate = comp_rates.mean()
    else:
        day_of_week = date_obj.weekday()
        weekly_avg = comp_prices[comp_prices['check_date'].dt.weekday == day_of_week]['price'].mean()
        competitor_rate = weekly_avg if not pd.isna(weekly_avg) else comp_prices['price'].mean()

    # Prophet prediction
    model = models[room_type]
    future = pd.DataFrame({'ds': [date_obj], 'competitor_price': [competitor_rate]})
    forecast = model.predict(future)
    prophet_price = forecast['yhat'].iloc[0]

    # Historical data for pricing
    historical_data = pd.read_csv("/data/bookings.csv")
    historical_prices = historical_data[historical_data['room_type'] == room_type]['price_per_day'].tolist()

    # Calculate final price
    result = future_price(room_type, occupancy, competitor_rate, historical_prices, date_obj)
    final_price = max(room_types[room_type]['floor'], min(room_types[room_type]['ceiling'], result['price']))

    return jsonify({
        'date': date,
        'room_type': room_type,
        'projected_price': round(final_price, 2),
        'reason': result['reason']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
