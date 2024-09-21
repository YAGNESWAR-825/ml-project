import streamlit as st
import pandas as pd
import pickle

# Load pre-trained model using pickle
model_filename = '/mnt/data/linear_regression_model.pkl'  # Replace with your model file path if different
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load dataset to extract feature columns
file_path = '/mnt/data/country_comparison_large_dataset.csv'  # Path to the uploaded dataset
data = pd.read_csv(file_path)

# Step 1: Extract feature columns (excluding the target column 'GDP (current US$)')
target_column = 'GDP (current US$)'  # Assuming 'GDP (current US$)' is the target column
feature_columns = [col for col in data.columns if col != target_column]

# Step 2: App Title
st.title("Economic Indicator Prediction App")

# Step 3: Input form for user to enter values for each feature
st.subheader("Enter the feature values for prediction:")

user_input = {}
for feature in feature_columns:
    user_input[feature] = st.number_input(f"Input value for {feature}", value=0.0)

# Step 4: Convert user inputs to DataFrame
input_df = pd.DataFrame([user_input])

# Step 5: Reorder input DataFrame to match the model's expected feature order
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# Step 6: Make predictions
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f"Predicted GDP: {prediction[0]}")
