import streamlit as st
import numpy as np
import pickle

# Load the trained Ridge model and scaler
with open('ridge_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("🚗 Car Price Predictor")
st.markdown("Enter the car details below to predict its price (in thousands ₹k).")

# ----------------------
# User Inputs
# ----------------------
mileage = st.slider("Mileage (km)", 0, 300000, 50000, step=1000)
engine_size = st.slider("Engine Size (L)", 0.8, 6.0, 2.0, step=0.1)
horsepower = st.slider("Horsepower", 50, 500, 100, step=1)
torque = st.slider("Torque", 50, 600, 150, step=1)
doors = st.selectbox("Doors", options=[2, 3, 4, 5, 6])
airbags = st.slider("Airbags", 0, 10, 4, step=1)
weight = st.slider("Weight (kg)", 800, 3000, 1500, step=10)
fuel_efficiency = st.slider("Fuel Efficiency (km/l)", 5.0, 40.0, 15.0, step=0.1)
brand_score = st.slider("Brand Score", 0.0, 10.0, 5.0, step=0.1)
luxury_index = st.slider("Luxury Index", 0.0, 10.0, 5.0, step=0.1)

# ----------------------
# Prediction
# ----------------------
if st.button("Predict Price"):
    # Apply log-transform to torque
    torque_log = np.log1p(torque)
    
    # Prepare input array
    input_data = np.array([[mileage, engine_size, horsepower, torque_log,
                            doors, airbags, weight, fuel_efficiency,
                            brand_score, luxury_index]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    price_pred = model.predict(input_scaled)[0]
    
    st.success(f"Predicted Car Price: ₹{price_pred:.2f}k")
