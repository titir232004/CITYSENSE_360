import streamlit as st
import os
import sys
import base64
import pandas as pd

# ---------------- PATH SETUP ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models import aqi
from models.complaints_analyser import process_complaint

st.set_page_config(page_title="CitySense360", layout="wide")
st.title("🌆 CitySense360 – Smart City Intelligence Dashboard")

# ---------------- BACKGROUND ----------------
def add_bg(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover; 
            background-attachment: fixed;
        }}
        html, body, [class*="st-"] {{ color: black !important; }}
        </style>
        """, unsafe_allow_html=True)

add_bg(os.path.join(APP_DIR, "bg.jpeg"))

# ---------------- AQI UTILS ----------------
def aqi_severity(aqi_val):
    if aqi_val <= 50: return "🟢 Good", "#2ecc71"
    elif aqi_val <= 100: return "🟡 Moderate", "#f1c40f"
    elif aqi_val <= 200: return "🟠 Poor", "#e67e22"
    else: return "🔴 Severe", "#e74c3c"

@st.cache_resource
def init_aqi():
    """Initialize AQI models (loads trained models)."""
    aqi.initialize_system()
    return True

aqi_loaded = init_aqi()

tab1, tab2 = st.tabs(["🌫 AQI Monitoring", "🗣 Citizen Complaint Analyzer"])

# ---------------- AQI TAB ----------------
with (tab1):
    if aqi_loaded and aqi.city_data:
        city = st.selectbox("Select City", list(aqi.city_data.keys()))
        df = aqi.city_data[city]

        # Current AQI
        current = aqi.get_current_aqi(city)

        # Next-day prediction
        next_day = aqi.predict_next_day(city)  # updated function in aqi.py

        # Health score & anomaly detection
        score, status = aqi.health_score(current)
        is_anomaly, diff = aqi.detect_anomaly(city)

        # Display metrics
        c1, c2, c3, c4 = st.columns(4)
        severity, color = aqi_severity(current)
        anomaly_text = f"⚠️ Anomaly Detected (Δ {diff:.1f})" if is_anomaly else "✅ No Anomaly"

        c1.markdown(f"""
        <div style="background:{color};padding:20px;border-radius:12px;text-align:center;">
        <h3>Current AQI</h3><h1>{current:.1f}</h1><h4>{severity}</h4></div>""", unsafe_allow_html=True)

        c2.metric("Next-Day AQI", f"{next_day:.1f}")
        c3.metric("City Health Score", f"{score} – {status}")
        c4.metric("Anomaly Status", anomaly_text)

        # Historical AQI plot
        st.subheader("📈 Historical AQI Trend")
        st.pyplot(aqi.plot_city_aqi(city))

        # Model evaluation
        rmse, mae = aqi.evaluate_model(city)
        st.info(f"Model Accuracy → RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    else:
        st.warning("⚠ AQI data not loaded or missing.")

# ---------------- COMPLAINT TAB ----------------
with tab2:
    complaint = st.text_area("Enter citizen complaint", height=150)
    if st.button("Analyze Complaint"):
        if not complaint.strip():
            st.warning("Please enter a complaint.")
        else:
            try:
                result = process_complaint(complaint)
                st.success("✅ Complaint Analyzed Successfully")
                st.write("**Summary:**", result["summary"])
                st.write("**Category:**", result["predicted_category"])
                st.write("**Routed Department:**", result["routed_department"])
            except Exception as e:
                st.exception(e)
