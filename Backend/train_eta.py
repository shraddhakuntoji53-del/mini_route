# backend/ml/train_eta.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Place a CSV at backend/ml/historical_trips.csv with columns:
# distance_km, time_of_day_hour, congestion_index, travel_time_seconds

csv_path = "backend/ml/historical_trips.csv"
model_path = "backend/models/eta_lr.pkl"

if not os.path.exists(csv_path):
    print("No historical CSV found at", csv_path)
    # create a tiny synthetic dataset for demo
    import numpy as np
    N = 500
    data = {
        "distance_km": np.random.uniform(0.5, 500, N),
        "time_of_day_hour": np.random.uniform(0,24,N),
        "congestion_index": np.random.uniform(0,80,N),
    }
    df = pd.DataFrame(data)
    # generate travel_time_seconds with noise
    df["travel_time_seconds"] = ((df["distance_km"] / (50 - df["congestion_index"]*0.3 + 1e-3)) * 3600).astype(int) + np.random.randint(-300,300,N)
    df.to_csv(csv_path, index=False)
else:
    df = pd.read_csv(csv_path)

X = df[["distance_km", "time_of_day_hour", "congestion_index"]]
y = df["travel_time_seconds"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train,y_train)
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print("Model trained and saved to", model_path)
