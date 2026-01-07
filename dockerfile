# ---------------- BASE IMAGE ----------------
FROM python:3.10-slim

# ---------------- ENV SETTINGS ----------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------------- WORKDIR ----------------
WORKDIR /app

# ---------------- SYSTEM DEPENDENCIES ----------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------- COPY REQUIREMENTS ----------------
COPY requirements.txt .

# ---------------- INSTALL PYTHON DEPS ----------------
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ---------------- COPY PROJECT FILES ----------------
# Copy Streamlit app
COPY app/streamlit_app.py ./streamlit_app.py

# Copy Assets folder (with capital A) into container
COPY Assets/ ./Assets/

COPY datasets/ ./datasets/
# Copy models folder
COPY models/ ./models/


# ---------------- STREAMLIT CONFIG ----------------
EXPOSE 8501

# ---------------- RUN APP ----------------
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--browser.serverAddress=localhost"]
