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

DISTRICTS = [
    "Lisboa", "Porto", "Setúbal", "Braga", "Faro", "Coimbra", "Aveiro",
    "Leiria", "Santarém", "Castelo Branco", "Viseu", "Viana do Castelo",
    "Évora", "Vila Real", "Beja", "Guarda", "Portalegre", "Bragança",
]
TOP_CITIES = [
    "Lisboa", "Sintra", "Porto", "Vila Nova de Gaia", "Cascais", "Loulé",
    "Braga", "Almada", "Oeiras", "Matosinhos", "Seixal", "Guimarães",
    "Coimbra", "Funchal", "Maia", "Gondomar", "Albufeira", "Setúbal",
]

col1, col2 = st.columns(2)

with col1:
    total_area = st.number_input("Total area (m²)", min_value=16.0, max_value=100000.0, value=100.0)
    living_area = st.number_input("Living area (m²)", min_value=0.0, max_value=100000.0, value=80.0)
    parking = st.number_input("Parking spaces", min_value=0.0, max_value=50.0, value=1.0)
    total_rooms = st.number_input("Total rooms", min_value=0.0, max_value=100.0, value=4.0)
    number_of_bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=50.0, value=2.0)
    construction_year = st.number_input("Construction year", min_value=1500.0, max_value=2031.0, value=2010.0)

with col2:
    district = st.selectbox("District", DISTRICTS)
    city = st.selectbox(
        "City", TOP_CITIES, index=0,
        help="The model learned price levels for every city with 30+ listings; "
             "rarer cities fall back to a pooled bucket.",
    )
    town = st.text_input(
        "Town / parish (optional)", value="", max_chars=120,
        help="Free text. Towns seen in training sharpen the estimate; unknown "
             "towns are handled gracefully.",
    )
    prop_type = st.selectbox(
        "Property type", ["Apartment", "House", "Duplex", "Studio", "Mansion", "Manor"]
    )
    energy_cert = st.selectbox(
        "Energy certificate", ["A+", "A", "B", "B-", "C", "D", "E", "F", "G", "NC"]
    )
    elevator = st.checkbox("Has elevator", value=False)

if st.button("Predict price"):
    if living_area > total_area:
        st.error("Living area cannot exceed total area.")
    else:
        input_df = pd.DataFrame([{
            "total_area": total_area,
            "parking": parking,
            "construction_year": construction_year,
            "total_rooms": total_rooms,
            "living_area": living_area,
            "number_of_bathrooms": number_of_bathrooms,
            "district": district,
            "city": city,
            "town": town,
            "type": prop_type,
            "energy_certificate": energy_cert,
            "elevator": elevator,
        }])
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated price: €{prediction:,.2f}")
