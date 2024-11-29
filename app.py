import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Load the model
model_path = "breast_cancer_model.h5"
model = load_model(model_path)

# Load the breast cancer dataset (for feature names)
data = load_breast_cancer()
feature_names = data.feature_names

# StandardScaler for input normalization
scaler = StandardScaler()
scaler.fit(data.data)  # Fit on the entire dataset for Streamlit app use

# Streamlit app UI
st.title("Breast Cancer Prediction using ANN")

# Input form for user data
st.header("Enter Features")
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

# Convert user input to a DataFrame for prediction
input_df = pd.DataFrame([user_input])
scaled_input = scaler.transform(input_df)

# Predict and display results
if st.button("Predict"):
    prediction_prob = model.predict(scaled_input)[0][0]
    prediction_class = "Malignant" if prediction_prob > 0.5 else "Benign"
    
    st.subheader("Prediction Results")
    st.write(f"**Prediction:** {prediction_class}")
    st.write(f"**Probability of Malignancy:** {prediction_prob:.4f}")


    # Streamlit app
st.title("Breast Cancer Prediction")
st.write("Enter the feature values to predict breast cancer.")

# User input form
user_input = {}
for feature in X.columns[selected_features]:
    user_input[feature] = st.number_input(f"Enter value for {feature}:", value=0.0)

# Predict button
if st.button("Predict"):
    # Load saved objects
    with open("feature_selector.pkl", "rb") as f:
        loaded_selector = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        loaded_scaler = pickle.load(f)

    with open("ann_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)

    # Preprocess user input
    input_data = pd.DataFrame([user_input])
    input_data = loaded_selector.transform(input_data)
    input_data = loaded_scaler.transform(input_data)

    # Make prediction
    prediction = loaded_model.predict(input_data)
    prediction_prob = loaded_model.predict_proba(input_data)[:, 1]

    # Display result
    if prediction[0] == 1:
        st.success(f"The model predicts that the patient has breast cancer with a probability of {prediction_prob[0]:.2f}.")
    else:
        st.success(f"The model predicts that the patient does not have breast cancer with a probability of {1 - prediction_prob[0]:.2f}.")
        
