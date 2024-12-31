import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('rf_model.pkl')
  #  Random Forest is used

# Load the dataset for feature reference
data = pd.read_csv("C:\\Users\\Saiteja\\Downloads\\car_data.csv")

# Preprocess categorical features
bike_names = [...]  # Same list as before
data = data[~data["Car_Name"].isin(bike_names)]

car_names = data['Car_Name'].unique()
fuel_types = data['Fuel_Type'].unique()
transmissions = data['Transmission'].unique()
owners = data['Owner'].unique()
selling_types = data['Selling_type'].unique()

# Streamlit UI
st.title("Car Price Prediction")
st.header("Input Car Features")

# User inputs
car_name = st.selectbox("Car Name", options=car_names)
year = st.number_input("Year", min_value=data['Year'].min(), max_value=data['Year'].max())
present_price = st.number_input("Present Price (in Lakh)", min_value=0.0, step=0.1)
driven_kms = st.number_input("Driven Kilometers", min_value=0)
fuel = st.selectbox("Fuel Type", options=fuel_types)
transmission = st.selectbox("Transmission", options=transmissions)
owner = st.selectbox("Owner", options=owners)
selling_type = st.selectbox("Selling Type", options=selling_types)

# Prepare input data
input_df = pd.DataFrame({
    'Car_Name': [car_name],
    'Year': [year],
    'Present_Price': [present_price],
    'Driven_kms': [driven_kms],
    'Fuel_Type': [fuel],
    'Transmission': [transmission],
    'Owner': [owner],
    'Selling_type': [selling_type]
})

# Encode input data
input_df = pd.get_dummies(input_df, columns=['Car_Name', 'Fuel_Type', 'Transmission', 'Owner', 'Selling_type'], drop_first=True)
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"The predicted price of the car is {max(0, prediction[0]):,.2f} Lakh INR")
