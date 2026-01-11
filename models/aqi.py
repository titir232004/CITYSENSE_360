# models/aqi.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")

SEQ_LEN = 30
BATCH_SIZE = 64
EPOCHS = 60
LR = 0.0007

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
FEATURES = POLLUTANTS + ["AQI"]
TARGET = "AQI"

# ================= STORAGE =================
city_data = {}   # city -> dataframe
scalers = {}     # city -> MinMaxScaler
models = {}      # city -> trained PyTorch model

# ================= MODEL =================
class AQILSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).squeeze()


# ================= DATA LOAD =================
def load_city_data():
    os.makedirs(MODEL_DIR, exist_ok=True)

    for file in os.listdir(DATA_DIR):
        if not file.endswith(".csv"):
            continue

        city = file.split("_")[0]  # handle cityname_AQI_DATASET.csv
        df = pd.read_csv(os.path.join(DATA_DIR, file))
        df.columns = df.columns.str.strip()

        # Ensure required columns exist
        if not all(c in df.columns for c in POLLUTANTS + ["AQI", "Date"]):
            print(f"[WARN] Skipping {city}: required columns missing")
            continue

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").dropna(subset=FEATURES + ["Date"])  # drop rows missing features or date

        if len(df) < SEQ_LEN:
            print(f"[WARN] Skipping {city}: not enough data after cleaning")
            continue

        scaler = MinMaxScaler()
        df[FEATURES] = scaler.fit_transform(df[FEATURES])

        city_data[city] = df
        scalers[city] = scaler
        print(f"[INFO] Loaded {city} with {len(df)} rows")


# ================= SEQUENCES =================
def make_sequences(data):
    X, y = [], []
    for i in range(len(data) - SEQ_LEN):
        X.append(data[i:i+SEQ_LEN])
        y.append(data[i+SEQ_LEN][FEATURES.index("AQI")])
    return np.array(X), np.array(y)

# ================= TRAIN =================
def train_city_model(city):
    df = city_data[city]
    vals = df[FEATURES].values

    X, y = make_sequences(vals)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                      torch.tensor(y, dtype=torch.float32)),
                        batch_size=BATCH_SIZE, shuffle=True)

    model = AQILSTM(input_size=len(FEATURES))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.HuberLoss()

    best_loss = float("inf")
    path = os.path.join(MODEL_DIR, f"{city}_aqi.pth")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), path)

    model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    model.eval()
    models[city] = model
    print(f"✅ Trained model saved for {city}")

# ================= INITIALIZE =================
def initialize_system():
    load_city_data()
    for city in city_data:
        path = os.path.join(MODEL_DIR, f"{city}_aqi.pth")
        model = AQILSTM(input_size=len(FEATURES))
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            models[city] = model
        else:
            train_city_model(city)

# ================= PREDICTIONS =================
def predict_next_day(city):
    df = city_data[city]
    seq = torch.tensor(df[FEATURES].values[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred_norm = models[city](seq).item()
    # Inverse scale AQI
    X_last = df[FEATURES].values[-1].reshape(1, -1)
    X_last[0, -1] = pred_norm
    pred_actual = scalers[city].inverse_transform(X_last)[0, -1]
    return pred_actual

# ================= METRICS =================
def evaluate_model(city):
    df = city_data[city]
    vals = df[FEATURES].values
    X, y = make_sequences(vals)
    with torch.no_grad():
        preds = models[city](torch.tensor(X, dtype=torch.float32)).numpy()

    # inverse transform
    preds_scaled = np.hstack([X[:, -1, :-1], preds.reshape(-1, 1)])
    y_scaled = np.hstack([X[:, -1, :-1], y.reshape(-1, 1)])
    inv_preds = scalers[city].inverse_transform(preds_scaled)[:, -1]
    inv_y = scalers[city].inverse_transform(y_scaled)[:, -1]

    rmse = mean_squared_error(inv_y, inv_preds) ** 0.5
    mae = mean_absolute_error(inv_y, inv_preds)
    return rmse, mae

# ================= HEALTH & ANOMALY =================
def health_score(aqi_val):
    if aqi_val <= 50: return "Healthy", "🟢"
    elif aqi_val <= 100: return "Moderate", "🟡"
    elif aqi_val <= 200: return "Poor", "🟠"
    else: return "Severe", "🔴"

def get_current_aqi(city):
    X_last = city_data[city][FEATURES].values[-1].reshape(1, -1)
    return float(scalers[city].inverse_transform(X_last)[0, -1])

def detect_anomaly(city, threshold=25):
    cur_aqi = get_current_aqi(city)
    nxt_aqi = predict_next_day(city)
    diff = abs(nxt_aqi - cur_aqi)
    return diff > threshold, diff

# ================= PLOTTING =================
def plot_city_aqi(city):
    df = city_data[city]
    # inverse scale AQI for plotting
    scaled = scalers[city].inverse_transform(df[FEATURES].values)
    aqi_vals = scaled[:, -1]
    plt.figure(figsize=(10, 4))
    plt.plot(df["Date"], aqi_vals, marker='o', linestyle='-', color='tab:blue')
    plt.title(f"{city} Historical AQI")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.grid(True)
    plt.tight_layout()
    return plt
