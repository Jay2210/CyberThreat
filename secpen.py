# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load("xgb_pipeline.joblib")
scaler = pipeline["scaler"]
model = pipeline["model"]
le = pipeline["label_encoder"]

FEATURES = list(scaler.feature_names_in_)

st.title("SecPen Cybersecurity Threat Classifier")
st.markdown("Upload a CSV of network flows to predict threat type.")

uploaded = st.file_uploader("Upload CSV file", type="csv")

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Drop unexpected columns
    extra = [c for c in df.columns if c not in FEATURES]
    if extra:
        st.warning(f"Dropping {len(extra)} unexpected columns")
        df = df.drop(columns=extra)

    # Add any missing features (fill with zero)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    return df[FEATURES]

if uploaded:
    df = pd.read_csv(uploaded)
    X = prepare(df)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    result = X.copy()
    result["Prediction"] = le.inverse_transform(preds)

    st.dataframe(result)
else:
    st.info("Please upload a CSV file containing the network flow features used for prediction.")
