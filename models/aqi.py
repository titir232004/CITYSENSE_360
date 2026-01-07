import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import streamlit as st  # for warnings/errors

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")

SEQ_LEN = 10

city_data = {}
scalers = {}
models = {}

# ---------------- LSTM MODEL ----------------
class AQILSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ---------------- LOAD DATA ----------------
def load_city_data():
    city_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_AQI_Dataset.csv")]

    for file in city_files:
        city = file.split("_")[0]
        path = os.path.join(DATA_DIR, file)

        try:
            df = pd.read_csv(path)
        except Exception as e:
            st.error(f"Failed to load {file}: {e}")
            continue

        df.columns = df.columns.str.strip()
        if "Date" not in df.columns or "PM2.5" not in df.columns:
            st.warning(f"Skipping {file} — missing 'Date' or 'PM2.5'")
            continue

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "PM2.5"])
        df = df.sort_values("Date")[["Date", "PM2.5"]]
        df["PM2.5"].ffill(inplace=True)

        scaler = MinMaxScaler()
        df["scaled"] = scaler.fit_transform(df[["PM2.5"]])

        city_data[city] = df
        scalers[city] = scaler

# ---------------- LOAD MODELS FROM DISK ----------------
def load_trained_models():
    """
    Loads pre-trained AQI LSTM models from models/ folder
    """
    for city in city_data.keys():
        model_file = os.path.join(MODEL_DIR, f"{city}_aqi_model.pth")
        if os.path.exists(model_file):
            model = AQILSTM()
            model.load_state_dict(torch.load(model_file))
            model.eval()
            models[city] = model
        else:
            st.warning(f"Pre-trained model not found for {city}, it will need training.")

# ---------------- SEQUENCE CREATOR ----------------
def create_sequences(data):
    X, y = [], []
    for i in range(len(data) - SEQ_LEN):
        X.append(data[i:i + SEQ_LEN])
        y.append(data[i + SEQ_LEN])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ---------------- CURRENT & NEXT DAY AQI ----------------
def get_current_aqi(city):
    df = city_data[city]
    return df.iloc[-1]["PM2.5"], df.iloc[-1]["Date"]

def predict_next_day_aqi(city):
    model = models[city]
    seq = torch.tensor(city_data[city]["scaled"].values[-SEQ_LEN:], dtype=torch.float32)
    with torch.no_grad():
        pred = model(seq.unsqueeze(0)).item()
    return scalers[city].inverse_transform([[pred]])[0][0]

# ---------------- PLOT FOR STREAMLIT ----------------
def plot_city_aqi(city):
    df = city_data[city]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["Date"], df["PM2.5"])
    ax.set_title(f"{city} PM2.5 Trend")
    ax.set_ylabel("PM2.5")
    ax.set_xlabel("Date")
    return fig

# ---------------- INIT (STREAMLIT SAFE) ----------------
@st.cache_resource
def initialize_aqi_system():
    """
    Loads city data and pre-trained models for Streamlit.
    Training is skipped if .pth models exist.
    """
    load_city_data()
    load_trained_models()
