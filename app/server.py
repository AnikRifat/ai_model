from prophet import Prophet
from flask import Flask, request, jsonify
import pandas as pd
import os
import pickle

app = Flask(__name__)
model_dir = "/app/models"
os.makedirs(model_dir, exist_ok=True)

# Room types
room_types = ['Super Deluxe Twin', 'Super Deluxe King', 'Infinity Sea View', 'Junior Suite', 'Panorama Ocean Suite']

# Holiday list for 2025 with descriptive names
holidays = pd.DataFrame({
    'holiday': [
        'Shab-e-Barat', 'Mother Language Day', 'Independence Day', 'Shab-e-Qadar', 'Jumatul Bidah',
        'Eid-ul-Fitr', 'Eid-ul-Fitr', 'Eid-ul-Fitr', 'Eid-ul-Fitr', 'Eid-ul-Fitr', 'Pahela Baishakh',
        'May Day', 'Buddha Purnima', 'Eid-ul-Azha', 'Eid-ul-Azha', 'Eid-ul-Azha', 'Eid-ul-Azha', 'Eid-ul-Azha',
        'Ashura', 'Janmashtami', 'Eid-e-Milad-un-Nabi', 'Durga Puja', 'Durga Puja', 'Bijoy Dibosh', 'Christmas'
    ],
    'ds': pd.to_datetime([
        '2025-02-15', '2025-02-21', '2025-03-26', '2025-03-28', '2025-03-28',
        '2025-03-29', '2025-03-30', '2025-03-31', '2025-04-01', '2025-04-02', '2025-04-14',
        '2025-05-01', '2025-05-11', '2025-06-05', '2025-06-06', '2025-06-07', '2025-06-08', '2025-06-09',
        '2025-07-06', '2025-08-16', '2025-09-05', '2025-10-01', '2025-10-02', '2025-12-16', '2025-12-25'
    ]),
    'lower_window': 0,
    'upper_window': 0,
})

# Train or load price models
price_models = {}
for room_type in room_types:
    model_path = f"{model_dir}/{room_type.replace(' ', '_')}_price_model.pkl"
    if not os.path.exists(model_path):
        bookings = pd.read_csv("/data/bookings.csv")
        df = bookings[bookings['room_type'] == room_type][['check_in_date', 'price_per_day']].rename(
            columns={'check_in_date': 'ds', 'price_per_day': 'y'}
        )
        df['ds'] = pd.to_datetime(df['ds'])

        comp_prices = pd.read_csv("/data/competitors_room_prices.csv")
        comp_prices = comp_prices[['check_date', 'price']].rename(
            columns={'check_date': 'ds', 'price': 'competitor_price'}
        )
        comp_prices['ds'] = pd.to_datetime(comp_prices['ds'])
        comp_prices = comp_prices.groupby('ds', as_index=False).agg({'competitor_price': 'mean'})

        df = pd.merge(df, comp_prices, on='ds', how='left').fillna({'competitor_price': comp_prices['competitor_price'].mean()})

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=holidays
        )
        model.add_regressor('competitor_price')
        model.fit(df)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    price_models[room_type] = model

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

    model = price_models[room_type]
    forecast = model.predict(future)
    prediction = forecast['yhat'].iloc[0]

    # Extract components
    trend = forecast['trend'].iloc[0]
    yearly = forecast['yearly'].iloc[0]
    weekly = forecast['weekly'].iloc[0]
    holiday_effect = forecast['holidays'].iloc[0] if 'holidays' in forecast and forecast['holidays'].iloc[0] != 0 else 0
    comp_effect = forecast['competitor_price'].iloc[0] if 'competitor_price' in forecast else 0

    # Historical average for reference (from training data)
    bookings = pd.read_csv("/data/bookings.csv")
    hist_avg = bookings[bookings['room_type'] == room_type]['price_per_day'].mean()

    # Generate human-readable reason
    reasons = []
    if holiday_effect > 500:  # Significant holiday impact
        holiday_name = holidays[holidays['ds'] == pd.to_datetime(date)]['holiday'].iloc[0] if pd.to_datetime(date) in holidays['ds'].values else "Holiday"
        reasons.append(f"Higher due to {holiday_name} holiday")
    elif holiday_effect < -500:
        reasons.append(f"Lower due to post-holiday drop")
    if yearly > 500:
        reasons.append("Higher due to peak season trend")
    elif yearly < -500:
        reasons.append("Lower due to off-season trend")
    if weekly > 200:
        reasons.append("Higher due to weekend demand")
    elif weekly < -200:
        reasons.append("Lower due to weekday lull")
    if comp_effect > 200:
        reasons.append("Higher due to competitor pricing")
    elif comp_effect < -200:
        reasons.append("Lower due to competitive pressure")
    if abs(prediction - hist_avg) < 500 and not reasons:
        reasons.append("Stable, close to historical average")

    reason = " and ".join(reasons) if reasons else "Stable with minor trend adjustments"

    return jsonify({
        'date': date,
        'room_type': room_type,
        'projected_price': round(prediction, 2),
        'reason': reason
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)