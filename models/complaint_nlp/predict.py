import pandas as pd
import pickle
import argparse
import os

def load_model_vectorizer(model_path, vectorizer_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


import os


def analyze_complaints(input_csv, model, vectorizer, output_csv):
    df = pd.read_csv(input_csv)

    if "complaint" not in df.columns:
        raise ValueError("Input CSV must contain a 'complaint' column")

    # Transform complaints
    X_vec = vectorizer.transform(df["complaint"])

    # Predict category
    df["category_pred"] = model.predict(X_vec)

    # Add placeholder columns
    for col in ["intent", "response", "sentiment", "urgency", "fraud"]:
        if col not in df.columns:
            df[col] = ""

    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save predictions
    df.to_csv(output_csv, index=False)
    print(f"✅ Analysis completed. Predictions saved to {output_csv}")


if __name__ == "__main__":
    model_path = r"E:\GUVI_FINAL_PROJECT\CitySense360\models\complaint_nlp\model.pkl"
    vectorizer_path = r"E:\GUVI_FINAL_PROJECT\CitySense360\models\complaint_nlp\vectorizer.pkl"
    input_csv = r"E:\GUVI_FINAL_PROJECT\CitySense360\data\raw\synthetic_complaints.csv"
    output_csv = r"E:\GUVI_FINAL_PROJECT\CitySense360\data\processed\complaints_analysis.csv"

    model, vectorizer = load_model_vectorizer(model_path, vectorizer_path)
    analyze_complaints(input_csv, model, vectorizer, output_csv)
