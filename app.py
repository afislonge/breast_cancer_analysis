import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import pickle
import warnings
warnings.filterwarnings("ignore")

# Streamlit app
st.title("Breast Cancer Prediction using ANN")
st.write("Enter the patient bresat scan feature values to predict breast cancer.")

feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension']

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
# Initialize the list of dictionaries
feature_info = []

# Loop through each feature in the DataFrame
for feature in X.columns:
    feature_info.append({
        'feature': feature,           # Feature name (same as label in this case)
        'min_val': X[feature].min(),  # Minimum value of the feature
        'max_val': X[feature].max()   # Maximum value of the feature
    })

# User input form
user_input = {}
for feature in feature_info:
    feature_label = feature['feature']
    feature_min = feature['min_val']
    feature_max = feature['max_val']
    user_input[feature_label] = st.slider(f"Enter value for {feature_label}:",
                                          feature_min,
                                          feature_max,
                                          value=float(0),
                                          step=0.01)
    #user_input[feature] = st.number_input(f"Enter value for {feature}:", value=0.0)

# Predict button
if st.button("Predict"):
    # Load saved objects
    with open("breast_cancer_analysis\model\\feature_selector.pkl", "rb") as f:
        loaded_selector = pickle.load(f)

    with open("breast_cancer_analysis\model\\scaler.pkl", "rb") as f:
        loaded_scaler = pickle.load(f)

    with open("breast_cancer_analysis\model\\ann_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
        
    with open("breast_cancer_analysis\model\\custom_ann_model.pkl", "rb") as f:
        loaded_custom_model = pickle.load(f)

    # Preprocess user input
    input_data = pd.DataFrame([user_input])
    input_data = loaded_scaler.transform(input_data)
    input_data = loaded_selector.transform(input_data)

    # Make prediction
    prediction = loaded_model.predict(input_data)
    prediction_prob = loaded_model.predict_proba(input_data)[:, 1]
    prediction_class = "Malignant" if prediction_prob > 0.5 else "Benign"

    # Make prediction with custom model
    custom_prediction = loaded_custom_model.predict(input_data)
    pred = custom_prediction[0]
    custom_prediction_prob = round(pred[0],2)
    custom_prediction_class = "Malignant" if custom_prediction_prob > 0.5 else "Benign"

    # Display result
    st.subheader("Breast Cancer Prediction Results")
    st.write(f"**Prediction:** {prediction_class}")
    if prediction[0] == 1:
        st.error(f"The model predicts that the patient has breast cancer with a probability of {prediction_prob[0]:.2f}.")
    else:
        st.success(f"The model predicts that the patient does not have breast cancer with a probability of {1 - prediction_prob[0]:.2f}.")


    st.subheader("Breast Cancer Prediction Results (Custom Model)")
    st.write(f"**Prediction:** {custom_prediction_class}")
    if custom_prediction_prob == 1:
        st.error(f"The custom model predicts that the patient has breast cancer with a probability of {custom_prediction_prob:.2f}.")
    else:
        st.success(f"The custom model predicts that the patient does not have breast cancer with a probability of {1 - custom_prediction_prob:.2f}.")

