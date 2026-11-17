
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "csvdata.csv")

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found: {DATA_PATH}")
        st.info("Please keep csvdata.csv inside the same folder as app.py")
        st.stop()

    return pd.read_csv(DATA_PATH)

df = load_data()


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / r"C:\Users\Admin\Downloads\Home_Price_Predictions(ML)\best_home_price_model.pkt"
DATA_PATH = BASE_DIR / r"C:\Users\Admin\Downloads\Home_Price_Predictions(ML)\csvdata.csv"


# ---------------------------------------------------------
# Load trained model and dataset
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    # Match the column name used while training the model
    data.columns = data.columns.str.strip()
    data = data.rename(columns={
        "No. of Bedrooms": "Bedrooms",
        "No.of Bedrooms": "Bedrooms"
    })

    # Remove exported index column if present
    data = data.drop(columns=["Unnamed: 0", "Unnamed:0"], errors="ignore")

    required_columns = ["City", "Area", "Location", "Bedrooms", "Price"]
    missing = [col for col in required_columns if col not in data.columns]

    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    data["Area"] = pd.to_numeric(data["Area"], errors="coerce")
    data["Bedrooms"] = pd.to_numeric(data["Bedrooms"], errors="coerce")
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")

    data = data.dropna(subset=required_columns).copy()

    for col in ["City", "Location"]:
        data[col] = data[col].astype(str).str.strip()

    return data


def format_rupees(value):
    """Format a number using Indian digit grouping."""
    return "₹" + f"{value:,.0f}"


# ---------------------------------------------------------
# Load files safely
# ---------------------------------------------------------
try:
    model = load_model()
    df = load_data()
except FileNotFoundError as error:
    st.error(
        "Required file not found. Keep these files in the same folder:\n\n"
        "- C:/Users/Admin/Downloads/Home_Price_Predictions(ML)/app.py\n"
        "- C:/Users/Admin/Downloads/Home_Price_Predictions(ML)/best_home_price_model.pkt\n"
        "- C:/Users/Admin/Downloads/Home_Price_Predictions(ML)/csvdata.csv"
    )
    st.exception(error)
    st.stop()
except Exception as error:
    st.error("Unable to load the model or dataset.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# App heading
# ---------------------------------------------------------
st.title("🏠 House Price Prediction")
st.write(
    "Enter the property details below to estimate the house price "
    "using a trained Machine Learning model."
)

st.divider()
# Dataset overview
col1, col2, col3 = st.columns(3)
col1.metric("Total Properties", f"{len(df):,}")
col2.metric("Cities Available", f"{df['City'].nunique():,}")
col3.metric("Locations Available", f"{df['Location'].nunique():,}")

st.divider()


# ---------------------------------------------------------
# Prediction form
# ---------------------------------------------------------
st.subheader("Enter House Details")

city_options = sorted(df["City"].dropna().unique().tolist())

with st.form("house_prediction_form"):
    city = st.selectbox("Select City", city_options)

    city_data = df[df["City"] == city]
    location_options = sorted(
        city_data["Location"].dropna().unique().tolist()
    )

    if not location_options:
        location_options = sorted(df["Location"].dropna().unique().tolist())

    location = st.selectbox("Select Location", location_options)

    left, right = st.columns(2)

    with left:
        area = st.number_input(
            "House Area (sq. ft.)",
            min_value=100,
            max_value=50000,
            value=1200,
            step=50
        )

    with right:
        bedroom_options = sorted(
            df["Bedrooms"].dropna().astype(int).unique().tolist()
        )
        bedrooms = st.selectbox(
            "Number of Bedrooms",
            bedroom_options,
            index=bedroom_options.index(2) if 2 in bedroom_options else 0
        )

    submitted = st.form_submit_button(
        "Predict House Price",
        use_container_width=True
    )


# ---------------------------------------------------------
# Prediction result
# ---------------------------------------------------------
if submitted:
    input_data = pd.DataFrame([{
        "City": city,
        "Area": area,
        "Location": location,
        "Bedrooms": bedrooms
    }])

    try:
        predicted_price = float(model.predict(input_data)[0])

        if predicted_price <= 0:
            st.warning(
                "The model returned a non-positive estimate. "
                "Please check the entered property details."
            )
        else:
            st.success("Prediction completed successfully!")

            st.markdown("### Estimated House Price")
            st.markdown(
                f"<h1 style='color:#16a34a;'>{format_rupees(predicted_price)}</h1>",
                unsafe_allow_html=True
            )

            result1, result2, result3 = st.columns(3)
            result1.metric("City", city)
            result2.metric("Area", f"{area:,} sq. ft.")
            result3.metric("Bedrooms", str(bedrooms))

            st.caption(
                "This is an ML-based estimate from the supplied dataset, "
                "not an official property valuation or market quote."
            )

    except Exception as error:
        st.error("Prediction failed. Please check the input details.")
        st.exception(error)
