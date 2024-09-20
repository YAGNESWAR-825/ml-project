import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pre-trained model
model_filename = 'linear_regression_model.pkl'  # Replace with your model file path if different
model = joblib.load(model_filename)

# Define the feature columns (replace with actual column names from your dataset)
feature_columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4']  # Replace with the actual feature names

# Step 1: App Title
st.title("Economic Indicator Prediction App")

# Step 2: Input form for user to enter values for each feature
st.subheader("Enter the feature values for prediction:")

user_input = {}
for feature in feature_columns:
    user_input[feature] = st.number_input(f"Input value for {feature}", value=0.0)

# Step 3: Convert user inputs to DataFrame
input_df = pd.DataFrame([user_input])

# Step 4: Make predictions
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f"Predicted output: {prediction[0]}")
