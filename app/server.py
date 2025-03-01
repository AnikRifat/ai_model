from prophet import Prophet
from flask import Flask, request, jsonify
import pandas as pd
import os
import pickle
import logging

app = Flask(__name__)
model_dir = "/app/models"
os.makedirs(model_dir, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Room types and realistic price ranges (RM)
room_types = {
    'Super Deluxe Twin': (600, 900),
    'Super Deluxe King': (700, 1000),
    'Infinity Sea View': (900, 1200),
    'Junior Suite': (1200, 1600),
    'Panorama Ocean Suite': (1600, 2000),
}

# Malaysian holidays for 2025
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

        # Log data for debugging
        logger.info(f"Training data for {room_type}: {df['y'].mean()} RM (mean), {df['y'].min()}–{df['y'].max()} RM (range)")

        comp_prices = pd.read_csv("/data/competitors_room_prices.csv")
        comp_prices = comp_prices[['check_date', 'price']].rename(
            columns={'check_date': 'ds', 'price': 'competitor_price'}
        )
        comp_prices['ds'] = pd.to_datetime(comp_prices['ds'])
        comp_prices = comp_prices.groupby('ds', as_index=False).agg({'competitor_price': 'mean'})

        df = pd.merge(df, comp_prices, on='ds', how='left').fillna({'competitor_price': comp_prices['competitor_price'].mean()})

        # Train Prophet model with constraints
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=holidays,
            changepoint_prior_scale=0.05,  # Reduce overfitting
            seasonality_prior_scale=10.0   # Smoother seasonality
        )
        model.add_regressor('competitor_price')
        model.fit(df)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    models[room_type] = model

@app.route('/predict', methods=['GET'])
def predict():
    date = request.args.get('date')
    room_type = request.args.get('room_type')
    if room_type not in room_types:
        return jsonify({'error': 'Invalid room type'}), 400

    comp_prices = pd.read_csv("/data/competitors_room_prices.csv")
    avg_comp_price = comp_prices['price'].mean()

    future = pd.DataFrame({
        'ds': [pd.to_datetime(date)],
        'competitor_price': [avg_comp_price]
    })

    model = models[room_type]
    forecast = model.predict(future)
    prediction = forecast['yhat'].iloc[0]

    # Constrain prediction to realistic range
    min_price, max_price = room_types[room_type]
    prediction = max(min_price, min(max_price, prediction))

    # Generate reason
    trend = forecast['trend'].iloc[0]
    yearly = forecast['yearly'].iloc[0]
    weekly = forecast['weekly'].iloc[0]
    holiday_effect = forecast['holidays'].iloc[0] if 'holidays' in forecast and forecast['holidays'].iloc[0] != 0 else 0
    comp_effect = forecast['competitor_price'].iloc[0] if 'competitor_price' in forecast else 0

    reasons = []
    if holiday_effect > 100:
        holiday_name = holidays[holidays['ds'] == pd.to_datetime(date)]['holiday'].iloc[0] if pd.to_datetime(date) in holidays['ds'].values else "Holiday"
        reasons.append(f"Higher due to {holiday_name}")
    elif holiday_effect < -100:
        reasons.append("Lower due to post-holiday drop")
    elif yearly > 100:
        reasons.append("Higher due to peak season trend")
    elif yearly < -100:
        reasons.append("Lower due to off-season trend")
    elif weekly > 50:
        reasons.append("Higher due to weekend demand")
    elif weekly < -50:
        reasons.append("Lower due to weekday lull")
    elif comp_effect > 50:
        reasons.append("Higher due to competitor pricing")
    elif comp_effect < -50:
        reasons.append("Lower due to competitive pressure")
    else:
        reasons.append("Stable with minor trend adjustments")

    reason = reasons[0]  # Pick the most significant reason

    return jsonify({
        'date': date,
        'room_type': room_type,
        'projected_price': round(prediction, 2),
        'reason': reason
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)