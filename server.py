from prophet import Prophet
from flask import Flask, request, jsonify
import pandas as pd
import os
import pickle

app = Flask(__name__)
model_path = "/app/model.pkl"

# Train or load model
if not os.path.exists(model_path):
    bookings = pd.read_csv("/data/bookings.csv")
    bookings = bookings[['check_in_date', 'price_per_day']].rename(
        columns={'check_in_date': 'ds', 'price_per_day': 'y'}
    )
    bookings['ds'] = pd.to_datetime(bookings['ds'])

    comp_prices = pd.read_csv("/data/competitors_room_prices.csv")
    comp_prices = comp_prices[['check_date', 'price']].rename(
        columns={'check_date': 'ds', 'price': 'competitor_price'}
    )
    comp_prices['ds'] = pd.to_datetime(comp_prices['ds'])
    comp_prices = comp_prices.groupby('ds', as_index=False).agg({'competitor_price': 'mean'})

    df = pd.merge(bookings, comp_prices, on='ds', how='left')
    df['competitor_price'] = df['competitor_price'].fillna(df['competitor_price'].mean())

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.add_regressor('competitor_price')
    model.fit(df)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
else:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

@app.route('/predict', methods=['GET'])
def predict():
    date = request.args.get('date')
    room_type = request.args.get('room_type')
    comp_prices = pd.read_csv("/data/competitors_room_prices.csv")
    avg_comp_price = comp_prices['price'].mean()
    
    future = pd.DataFrame({
        'ds': [pd.to_datetime(date)],
        'competitor_price': [avg_comp_price]
    })
    
    forecast = model.predict(future)
    prediction = forecast['yhat'].iloc[0]
    
    return jsonify({
        'date': date,
        'room_type': room_type,
        'projected_price': round(prediction, 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)