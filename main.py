from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Load model, scaler, and encoders
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Load feature order (from training)
# You can save this list during training as well
FEATURE_ORDER = [
    'transaction type', 'merchant_category', 'amount (INR)',
    'sender_age_group', 'receiver_age_group', 'sender_state',
    'sender_bank', 'receiver_bank', 'device_type', 'network_type',
    'hour_of_day', 'day_of_week', 'is_weekend'
]

CATEGORICAL_COLS = list(label_encoders.keys())

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])

    try:
        # Encode categorical columns
        for col in CATEGORICAL_COLS:
            if col in input_df.columns:
                input_df[col] = label_encoders[col].transform(input_df[col])

        # Ensure column order
        input_df = input_df[FEATURE_ORDER]

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        return jsonify({
            'prediction': 'FRAUD' if pred == 1 else 'LEGIT',
            'fraud_probability': round(prob, 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
