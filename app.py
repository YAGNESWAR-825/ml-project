import streamlit as st
import joblib
import pandas as pd

# Step 1: Load the saved linear regression model
model_filename = 'linear_regression_model.pkl'  # The saved model file
model = joblib.load(model_filename)

# Step 2: Load the dataset to get column names (you provided a file earlier, we'll assume it's already uploaded)
data_path = 'country_comparison_large_dataset.csv'  # Adjust the path as needed
data = pd.read_csv(data_path)

# Exclude the target column (GDP) and keep only the feature columns
target_column = 'GDP (current US$)'  # Adjust the name if needed
feature_columns = [col for col in data.columns if col != target_column]

# Step 3: Create the Streamlit app
st.title("GDP Prediction Model Deployment")

# Step 4: Define input fields for the features based on the dataset columns
st.header("Input Feature Values")

# Create input fields for each feature dynamically
user_input = {}
for feature in feature_columns:
    user_input[feature] = st.number_input(f'Enter value for {feature}', value=0.0)

# Step 5: Convert user input to a DataFrame
input_df = pd.DataFrame([user_input])

# Step 6: Make predictions
if st.button('Predict'):
    # Ensure that the input DataFrame has the correct order of columns as expected by the model
    input_df = input_df.reindex(columns=feature_columns)

    # Make predictions
    prediction = model.predict(input_df)[0]  # Predict GDP value

    # Display the prediction result
    st.subheader("Prediction Result:")
    st.write(f"The predicted GDP is: ${prediction:,.2f}")

# Optional: Display information about the feature inputs
if st.checkbox('Show Feature Input Information'):
    st.write("User input values:", user_input)
