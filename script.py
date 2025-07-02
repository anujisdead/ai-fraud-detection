import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("upi_transactions_2024.csv")  # Replace with your actual file path

# Drop irrelevant columns
df.drop(columns=['transaction id', 'timestamp', 'transaction_status'], inplace=True)

# Identify categorical columns
categorical_cols = [
    'transaction type', 'merchant_category', 'sender_age_group',
    'receiver_age_group', 'sender_state', 'sender_bank',
    'receiver_bank', 'device_type', 'network_type', 'day_of_week'
]

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target
X = df.drop('fraud_flag', axis=1)
y = df['fraud_flag']

# Scale numeric columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train logistic regression model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# --- Prediction Function ---

def predict_transaction(transaction_dict):
    input_df = pd.DataFrame([transaction_dict])

    # Encode categorical values
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = label_encoders[col].transform(input_df[col])

    # Ensure columns are in the same order
    input_df = input_df[X.columns]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    print("Prediction:", "FRAUD" if pred == 1 else "LEGIT", f"(probability: {prob:.2f})")

# --- Example Usage ---

sample_transaction = {
    'transaction type': 'P2P',
    'merchant_category': 'Food',
    'amount (INR)': 600,
    'sender_age_group': '26-35',
    'receiver_age_group': '18-25',
    'sender_state': 'Delhi',
    'sender_bank': 'Axis',
    'receiver_bank': 'SBI',
    'device_type': 'Android',
    'network_type': '4G',
    'hour_of_day': 10,
    'day_of_week': 'Monday',
    'is_weekend': 0
}

predict_transaction(sample_transaction)
import pickle

# Save model
with open('fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
