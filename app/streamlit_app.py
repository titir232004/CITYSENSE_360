import streamlit as st
import sys
import os

# ---------------- BASE DIR SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.aqi import (
    initialize_aqi_system,
    plot_city_aqi,
    get_current_aqi,
    predict_next_day_aqi,
)
from models.complaints_analyser import process_complaint

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config("CitySense360", layout="wide")
st.title("🌆 CitySense360 Smart City Dashboard")

# ---------------- LOAD AQI DATA (STREAMLIT SAFE) ----------------
@st.cache_resource
def load_aqi():
    try:
        initialize_aqi_system()
        return True
    except KeyError as e:
        st.error(f"AQI data could not be loaded: {e}")
        return False
    except Exception as e:
        st.error(f"Unexpected error loading AQI data: {e}")
        return False

aqi_loaded = load_aqi()

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["🌫 AQI Monitor", "🗣 Complaint Analyzer"])

from models import aqi

# ---------------- AQI TAB ----------------
with tab1:
    if aqi_loaded and len(aqi.city_data.keys()) > 0:
        available_cities = list(aqi.city_data.keys())
        city = st.selectbox("Select City", available_cities)

        try:
            current_aqi, date = aqi.get_current_aqi(city)
            next_aqi = aqi.predict_next_day_aqi(city)

            col1, col2 = st.columns(2)
            col1.metric("Current AQI (PM2.5)", f"{current_aqi:.2f}")
            col2.metric("Next Day Predicted AQI", f"{next_aqi:.2f}")

            st.pyplot(aqi.plot_city_aqi(city))

        except KeyError:
            st.warning(f"No AQI model or data found for {city}.")
        except Exception as e:
            st.error(f"Error while displaying AQI for {city}: {e}")

    else:
        st.info("AQI data not loaded or no valid cities found. Check datasets/models folder.")


# ---------------- COMPLAINT TAB ----------------
with tab2:
    text = st.text_area("Enter citizen complaint")

    if st.button("Analyze Complaint"):
        try:
            result = process_complaint(text)

            st.success("Analysis Complete")
            st.write("**Summary:**", result["summary"])
            st.write("**Category:**", result["predicted_category"])
            st.write("**Routed To:**", result["routed_department"])
        except Exception as e:
            st.error(f"Could not process complaint: {e}")
