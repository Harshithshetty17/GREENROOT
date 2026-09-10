import os
import joblib
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="AI Precision Crop Recommender", page_icon="🌱", layout="wide")

# 1. Load Trained Model Artifacts
@st.cache_resource
def load_models():
    model_path = os.path.join("models", "stacking_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    classes_path = os.path.join("models", "class_names.pkl")
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(classes_path)):
        return None, None, None
    return joblib.load(model_path), joblib.load(scaler_path), joblib.load(classes_path)

# 2. Regional District Fallback
@st.cache_data
def load_nfsm_districts():
    nfsm_path = os.path.join("data", "Cleaned_NFSM_Dataset.csv")
    if os.path.exists(nfsm_path):
        try:
            df_nfsm = pd.read_csv(nfsm_path)
            for col in ['District', 'district', 'District Name', 'taluku']:
                if col in df_nfsm.columns:
                    return sorted(df_nfsm[col].dropna().unique().tolist()[:50])
        except Exception:
            pass
    return ["Udupi", "Dakshina Kannada", "Bengaluru Rural", "Mysuru", "Dharwad"]

model, scaler, classes = load_models()
district_list = load_nfsm_districts()

if model is None:
    st.error("Model artifacts missing. Run 'python train.py' first.")
    st.stop()

# 3. Weather Fetching Helper
def get_live_weather(city, key):
    if not key:
        return None, "API Key missing."
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get("cod") == 200:
            return {
                "temp": res["main"]["temp"],
                "humidity": res["main"]["humidity"],
                "rainfall": res.get("rain", {}).get("1h", 0.0) * 24 * 30
            }, None
        return None, res.get("message", "Location not found.")
    except Exception as e:
        return None, str(e)

# --- UI Layout ---
st.title("🌱 AI-Driven Precision Crop Recommendation Platform")
st.markdown("**Multi-Model Stacking Ensemble** with **Live Weather Ingestion** and **Explainable AI (XAI)**")
st.write("---")

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("1. Location & Climate Ingestion")
    c1, c2 = st.columns(2)
    with c1:
        city_query = st.text_input("City / District", value="Udupi")
    with c2:
        api_key_query = st.text_input("OpenWeatherMap Key (Optional)", type="password")

    temp_val, hum_val, rain_val = 26.0, 78.0, 190.0

    if st.button("Fetch Live Weather"):
        if api_key_query:
            w_data, err = get_live_weather(city_query, api_key_query)
            if w_data:
                temp_val = float(w_data["temp"])
                hum_val = float(w_data["humidity"])
                rain_val = float(w_data["rainfall"])
                st.success(f"Retrieved: {temp_val:.1f}°C, {hum_val:.0f}% Humidity")
            else:
                st.warning(f"Could not retrieve: {err}. Using default sliders.")
        else:
            st.info("No API key entered. Adjust sliders manually below.")

    st.subheader("2. Soil Chemical Parameters")
    selected_district = st.selectbox("Regional Soil Baseline (Fallback):", ["Manual Input"] + district_list)

    col_n, col_p = st.columns(2)
    with col_n:
        n_in = st.number_input("Nitrogen (N) kg/ha", 0.0, 200.0, 60.0)
    with col_p:
        p_in = st.number_input("Phosphorus (P) kg/ha", 0.0, 200.0, 40.0)

    col_k, col_ph = st.columns(2)
    with col_k:
        k_in = st.number_input("Potassium (K) kg/ha", 0.0, 250.0, 45.0)
    with col_ph:
        ph_in = st.number_input("Soil pH", 3.0, 11.0, 6.2)

    temp_in = st.slider("Temperature (°C)", 5.0, 50.0, temp_val)
    hum_in = st.slider("Relative Humidity (%)", 10.0, 100.0, hum_val)
    rain_in = st.slider("Rainfall (mm)", 10.0, 350.0, rain_val)

    btn_recommend = st.button("🚀 Recommend Optimal Crops", type="primary", use_container_width=True)

with col_right:
    st.subheader("3. Model Decision & Suitability Ranking")
    if btn_recommend:
        raw = np.array([[n_in, p_in, k_in, temp_in, hum_in, ph_in, rain_in]])
        scaled = scaler.transform(raw)
        probs = model.predict_proba(scaled)[0]
        top_idx = np.argsort(probs)[-3:][::-1]

        best_crop = classes[top_idx[0]]
        st.success(f"### Recommended Primary Crop: **{best_crop.upper()}** ({probs[top_idx[0]]*100:.1f}% Confidence)")

        st.write("#### Top-3 Ranked Crops:")
        for rank, idx in enumerate(top_idx, 1):
            st.write(f"**{rank}. {classes[idx].capitalize()}** — {probs[idx]*100:.2f}% match")
            st.progress(float(probs[idx]))

        st.write("---")
        st.subheader("4. Local Feature Attribution (Explainability)")
        feat_labels = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        z_scores = scaled[0]

        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.barh(feat_labels, z_scores, color=['#2ca02c' if z >= 0 else '#d62728' for z in z_scores])
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel("Input Deviation from Benchmark Mean (Z-Score)")
        ax.set_title(f"Factors Driving {best_crop.capitalize()} Recommendation")
        st.pyplot(fig)
        st.caption("Green: Above-average condition | Red: Below-average condition")

        st.write("---")
        st.subheader("5. Actionable Soil Advisory")
        if best_crop == "rice" and rain_in < 150:
            st.info("💡 **Water Management:** Rice requires standing moisture. Supplement with canal or borewell irrigation.")
        elif best_crop != "rice" and rain_in > 220:
            st.info("💡 **Drainage Advisory:** High rainfall detected. Ensure drainage runoff channels to prevent root rot.")
        else:
            st.info(f"💡 Parameters align with physiological thresholds for **{best_crop.capitalize()}** cultivation.")
    else:
        st.info("Configure parameters on the left and click 'Recommend Optimal Crops'.")