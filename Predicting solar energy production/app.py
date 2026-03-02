import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Solar Energy Production Predictor",
    page_icon="☀️",
    layout="wide"
)

st.title("☀ Solar Energy Production Prediction App")
st.markdown("---")

# ---------------- LOAD FILES ---------------- #
@st.cache_resource
def load_files():
    model = joblib.load("solar_Energy_production_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("label_encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, encoders, feature_columns

model, scaler, label_encoders, feature_columns = load_files()

# ---------------- INPUT SECTION ---------------- #
col1, col2 = st.columns(2)

with col1:
    utility = st.selectbox("Utility", label_encoders["Utility"].classes_)
    city = st.selectbox("City/Town", label_encoders["City/Town"].classes_)
    county = st.selectbox("County", label_encoders["County"].classes_)
    zip_code = st.number_input("Zip", value=10001)

with col2:
    developer = st.selectbox("Developer", label_encoders["Developer"].classes_)
    metering = st.selectbox("Metering Method", label_encoders["Metering Method"].classes_)
    storage_size = st.number_input("Energy Storage System Size (kWac)", min_value=0.0, value=0.0)

# ---------------- PREDICTION ---------------- #
if st.button("Predict"):

    # Create input dataframe
    input_df = pd.DataFrame({
        "Utility": [utility],
        "City/Town": [city],
        "County": [county],
        "Zip": [zip_code],
        "Developer": [developer],
        "Metering Method": [metering],
        "Energy Storage System Size (kWac)": [storage_size]
    })

    # Create Has_Storage if model used it
    if "Has_Storage" in feature_columns:
        input_df["Has_Storage"] = 1 if storage_size > 0 else 0

    # Encode categorical features
    for col, encoder in label_encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = encoder.transform(input_df[col])
            except:
                input_df[col] = 0

    # Ensure exact same training feature order
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Scale ONLY numeric columns used during training
    num_cols = scaler.feature_names_in_
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Predict (log scale)
    log_prediction = model.predict(input_df)[0]

    # Convert back from log
    prediction = np.expm1(log_prediction)

    # ---------------- OUTPUT ---------------- #
    st.success(f"Predicted Annual PV Energy Production: {prediction:,.0f} kWh")

    colA, colB = st.columns(2)

    with colA:
        st.metric("Daily Production", f"{prediction/365:,.0f} kWh/day")

    with colB:
        st.metric("Monthly Production", f"{prediction/12:,.0f} kWh/month")

    # ---------------- FEATURE IMPORTANCE ---------------- #
    st.markdown("### 📊 Feature Importance")

    importance = model.feature_importances_

    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(feature_columns, importance)
    ax.invert_yaxis()
    ax.set_title("Feature Importance")
    st.pyplot(fig)

st.markdown("---")
st.markdown("© Solar Energy AI Prediction System")




