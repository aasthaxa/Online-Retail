import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Customer Repurchase Predictor",
    page_icon="🛍️",
    layout="centered"
)

@st.cache_resource
def load_model_artifact():
    try:
        return joblib.load("customer_repurchase_model.joblib")
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure 'customer_repurchase_model.joblib' is in the same directory.")
        return None

artifact = load_model_artifact()

if artifact:
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    threshold = artifact["threshold"]
    model_name = artifact["model_name"]

    st.title("🛍️ 90-Day Customer Repurchase Predictor")
    st.write(f"Predicting whether a customer will return to purchase within next 90 days using the trained **{model_name}** model.")

    st.markdown("### Customer Behavioral Features")

    col1, col2 = st.columns(2)

    with col1:
        frequency = st.number_input("Order Frequency (Total distinct orders)", min_value=1, value=5, step=1)
        monetary = st.number_input("Monetary Value ($ Total spent)", min_value=0.01, value=500.0, step=10.0)
        total_units = st.number_input("Total Units Purchased (Quantity)", min_value=1, value=250, step=5)
        total_lines = st.number_input("Total Invoice Lines (Distinct items bought across orders)", min_value=1, value=40, step=1)
        active_days = st.number_input("Active Days (Unique calendar days of buying)", min_value=1, value=5, step=1)

    with col2:
        recency = st.number_input("Recency (Days since last purchase)", min_value=0.0, value=25.0, step=1.0)
        tenure = st.number_input("Tenure (Days since first purchase)", min_value=0.0, value=300.0, step=1.0)
        avg_unit_price = st.number_input("Average Unit Price ($)", min_value=0.01, value=3.5, step=0.1)
        avg_line_rev = st.number_input("Average Invoice Line Revenue ($)", min_value=0.01, value=20.0, step=0.5)
        country = st.selectbox("Customer Country", ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands", "Belgium", "Switzerland", "Portugal", "Australia", "Other"])

    if st.button("🔮 Predict Repurchase Probability", type="primary"):
        # Construct features dictionary
        features = {
            "Frequency": frequency,
            "Monetary": monetary,
            "Total_Units": total_units,
            "Total_Lines": total_lines,
            "Active_Days": active_days,
            "Average_Unit_Price": avg_unit_price,
            "Average_Line_Revenue": avg_line_rev,
            "Recency": recency,
            "Tenure_Days": tenure,
            "Average_Order_Value": monetary / frequency if frequency > 0 else 0,
            "Average_Lines_Per_Order": total_lines / frequency if frequency > 0 else 0,
            "Country": country
        }

        # Process features
        df_row = pd.DataFrame([features])
        # Handle categorical encoding matching training columns
        df_row = pd.get_dummies(df_row, columns=["Country"], drop_first=True, dummy_na=True)
        
        # Explicitly align columns with training features and set missing ones to 0
        df_row = df_row.reindex(columns=feature_columns, fill_value=0)

        # Convert all columns to floats/ints to prevent bool issues inside scikit-learn transformers
        df_row = df_row.astype(float)

        # Predict probability
        proba = float(model.predict_proba(df_row)[:, 1][0])
        is_repurchase = proba >= threshold

        # Render results
        st.markdown("--- ")
        st.subheader("Prediction Results:")

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(label="Repurchase Probability", value=f"{proba:.1%}")
        with metric_col2:
            if is_repurchase:
                st.success("🔥 Likely to return! (Class: YES)")
            else:
                st.warning("💤 High risk of churn / idle (Class: NO)")

        st.progress(proba)
