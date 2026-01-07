import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# -------------------------------------------------
# PATH CONFIG (ROOT SAFE)
# -------------------------------------------------
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_DIR)

DATA_PATH = os.path.join(
    PROJECT_ROOT, "datasets", "citizen_grievances.csv"
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "complaint_model.pkl")

print("Looking for dataset at:", DATA_PATH)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# -------------------------------------------------
# 1. TRAIN COMPLAINT CLASSIFIER
# -------------------------------------------------
def train_model(csv_path):

    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    print("Columns:", df.columns.tolist())
    print("Total rows:", len(df))

    # Combine useful text fields
    df["text"] = (
        df["Category"].fillna("") + " " +
        df["Sub Category"].fillna("") + " " +
        df["Staff Remarks"].fillna("")
    )

    TEXT_COL = "text"
    LABEL_COL = "Category"

    df = df[[TEXT_COL, LABEL_COL]].dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COL],
        df[LABEL_COL],
        test_size=0.2,
        random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    print("Training complaint classifier...")
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"Complaint Classifier Accuracy: {acc:.2f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model saved at:", MODEL_PATH)


# -------------------------------------------------
# 2. COMPLAINT SUMMARIZATION (LLM-READY)
# -------------------------------------------------
def summarize_complaint(text, max_words=20):
    words = text.split()
    return " ".join(words[:max_words])


# -------------------------------------------------
# 3. DEPARTMENT ROUTING
# -------------------------------------------------
DEPARTMENT_MAP = {
    "garbage": "Municipal Corporation",
    "solid waste": "Municipal Corporation",
    "waste": "Municipal Corporation",
    "road": "Public Works Department",
    "pothole": "Public Works Department",
    "traffic": "Traffic Police",
    "water": "Water Supply Board",
    "sewage": "Water & Sewerage Board",
    "electricity": "Electricity Board",
    "pollution": "Pollution Control Board"
}

def route_complaint(text):
    text = text.lower()
    for keyword, dept in DEPARTMENT_MAP.items():
        if keyword in text:
            return dept
    return "General Administration"


# -------------------------------------------------
# 4. FULL PIPELINE (SAFE MODEL LOADING)
# -------------------------------------------------
def process_complaint(text):

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not trained yet. Run train_model() first.")

    model = joblib.load(MODEL_PATH)

    category = model.predict([text])[0]
    summary = summarize_complaint(text)
    department = route_complaint(text)

    return {
        "original_text": text,
        "summary": summary,
        "predicted_category": category,
        "routed_department": department
    }


# -------------------------------------------------
# 5. RUN SCRIPT
# -------------------------------------------------
if __name__ == "__main__":

    # Train model
    train_model(DATA_PATH)

    # Test example
    sample_text = (
        "Garbage has not been collected for several days "
        "and the area has become unhygienic."
    )

    result = process_complaint(sample_text)

    print("\nComplaint Analysis Result:")
    for k, v in result.items():
        print(f"{k}: {v}")

