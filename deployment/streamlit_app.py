"""
streamlit_app.py
----------------
Minimal Streamlit prototype for Portugal Housing Price Prediction.

Run: streamlit run streamlit_app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent / "model.pkl")


model = load_model()

st.set_page_config(page_title="Portugal Housing Price Predictor", layout="centered")
st.title("Portugal Housing Price Prediction")
st.write("Enter property features to get a price estimate in Euros.")

col1, col2 = st.columns(2)

with col1:
    total_area = st.number_input("Total Area (m²)", min_value=5.0, max_value=5000.0, value=100.0)
    living_area = st.number_input("Living Area (m²)", min_value=5.0, max_value=5000.0, value=80.0)
    parking = st.number_input("Parking Spaces", min_value=0.0, max_value=10.0, value=1.0)
    total_rooms = st.number_input("Total Rooms", min_value=0.0, max_value=30.0, value=4.0)
    number_of_bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=15.0, value=2.0)
    construction_year = st.number_input("Construction Year", min_value=1800.0, max_value=2026.0, value=2010.0)

with col2:
    district = st.selectbox("District", [
        "Lisboa", "Porto", "Setúbal", "Braga", "Faro", "Coimbra",
        "Aveiro", "Leiria", "Santarém", "Castelo Branco",
    ])
    prop_type = st.selectbox("Property Type", ["Apartment", "House", "Land", "Store", "Farm"])
    energy_cert = st.selectbox("Energy Certificate", ["A+", "A", "B", "B-", "C", "D", "E", "F", "NC"])
    elevator = st.checkbox("Has Elevator", value=False)

if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "total_area": total_area,
        "parking": parking,
        "construction_year": construction_year,
        "total_rooms": total_rooms,
        "living_area": living_area,
        "number_of_bathrooms": number_of_bathrooms,
        "district": district,
        "city": district,
        "town": "",
        "type": prop_type,
        "energy_certificate": energy_cert,
        "elevator": elevator,
    }])
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Price: €{prediction:,.2f}")
