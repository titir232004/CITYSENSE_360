# 🌆 CitySense360 – Smart City Analytics Dashboard

CitySense360 is an intelligent Smart City Analytics Dashboard that provides real-time insights into Air Quality Index (AQI) and citizen grievances across major cities.
The project combines data analysis, machine learning models, and an interactive Streamlit dashboard, all packaged inside a Docker container for easy deployment.
---

## 📁 Project Structure

CitySense360
```
|── app/
│    └── streamlit_app.py          # Main Streamlit dashboard
│
├── Assets/
│   └── bg.jpeg               # Background image
│
├── datasets/
│   ├── Mumbai_AQI_Dataset.csv
│   ├── Delhi_AQI_Dataset.csv
│   ├── Bangalore_AQI_Dataset.csv
│   ├── Chennai_AQI_Dataset.csv
│   ├── Hyderabad_AQI_Dataset.csv
│   └── citizen_grievances.csv    # Complaints dataset
│
├── models/
│   ├── aqi_model.pkl             # AQI prediction model
│   └── complaints_analyser.pkl       # Complaint classification model
│
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
└── README.md                     # Project documentation

```

## ⚙️ Features

✅ **Air Quality Index (AQI) Monitoring**  
1.Displays current AQI levels for selected cities
2.Color-coded AQI status (Good, Moderate, Poor, etc.)

✅ **Next-Day AQI Prediction**
1. Uses a trained ML model to predict future AQI values
   
✅ **Citizen Complaint Analyzer**  
1.Analyzes and categorizes public complaints
2.Helps identify major urban issues

✅ **Interactive Streamlit Dashboard**  
1.User-friendly UI
2.Dynamic city selection
3.Visual indicators and metrics

✅ **Dockerized Deployment**  
1.Fully containerized for consistent execution
2.Runs seamlessly across systems
---

## 🧩 Tech Stack

- **Python**
- **Pandas, NumPy** – Data handling  
- **Scikit-learn** – Machine learning models
- **Streamlit** –Interactive dashboard
- **Matplotlib / Plotly** – Data visualization
- **Docker** – Containerization  

---

## 🚀 How to Run

### 1️⃣ Clone the repository
```
git clone https://github.com/your-username/CitySense360.git
cd CitySense360
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the Application Locally
streamlit run app/streamlit_app.py
Open in browser: http://localhost:8501

### 🐳 Run Using Docker
1️⃣ Build Docker Image
docker build -t citysense360 .

2️⃣ Run Docker Container
docker run -p 8501:8501 citysense360

3️⃣ Access the Dashboard
http://localhost:8501
```
