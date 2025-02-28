import h2o
from h2o.automl import H2OAutoML
from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

# Initialize H2O
h2o.init()

# Load or train model
model_path = "/app/trained_model"
if not os.path.exists(model_path):
    data = h2o.import_file("/data/bookings.csv")
    aml = H2OAutoML(max_runtime_secs=300, seed=42)
    aml.train(y="sales_price", training_frame=data)
    h2o.save_model(aml.leader, path=model_path)
else:
    model = h2o.load_model(model_path)

@app.route('/predict', methods=['GET'])
def predict():
    date = request.args.get('date')
    room_type = request.args.get('room_type')

    future_data = pd.DataFrame({
        'check_in_date': [date],
        'room_type': [room_type],
    })
    h2o_frame = h2o.H2OFrame(future_data)

    prediction = model.predict(h2o_frame).as_data_frame().iloc[0, 0]

    return jsonify({
        'date': date,
        'room_type': room_type,
        'projected_price': round(prediction, 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)