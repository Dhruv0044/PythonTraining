import streamlit as st
import pickle
import pandas as pd

# Load the trained model pipeline
try:
    with open('final_pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'final_pipe.pkl' not found. Please run train.py first.")
    st.stop()

# Load the feature data for dropdowns
try:
    with open('data_features.pkl', 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    st.error("Data file 'data_features.pkl' not found. Please run train.py first.")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title('🚗 Car Price Predictor')
st.markdown("Enter the car details to get an estimated price.")

# Create columns for layout
col1, col2 = st.columns(2)

with col1:
    # Company selection
    company = st.selectbox(
        'Select Company',
        sorted(data['company'].unique())
    )

    # Model selection (filtered by company)
    available_models = sorted(data[data['company'] == company]['name'].unique())
    name = st.selectbox(
        'Select Model',
        available_models
    )

    # Fuel type selection
    fuel_type = st.selectbox(
        'Select Fuel Type',
        sorted(data['fuel_type'].unique())
    )

with col2:
    # Year selection
    year = st.number_input(
        'Enter Year of Manufacture',
        min_value=int(data['year'].min()),
        max_value=int(data['year'].max()),
        value=2015,
        step=1
    )

    # Kms driven input
    kms_driven = st.number_input(
        'Enter Kilometers Driven',
        min_value=0,
        value=50000,
        step=1000
    )

# --- Prediction Logic ---
if st.button('Predict Price', use_container_width=True):
    try:
        # Create a DataFrame from the inputs
        input_data = pd.DataFrame(
            [[name, company, fuel_type, year, kms_driven]],
            columns=['name', 'company', 'fuel_type', 'year', 'kms_driven']
        )

        # Make prediction
        prediction = pipe.predict(input_data)

        # Display the result
        st.success(f'### 💰 Estimated Price: ₹ {prediction[0]:,.2f}')

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")