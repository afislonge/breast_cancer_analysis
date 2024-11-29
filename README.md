# Breast Cancer Analysis and Prediction App

## Project Overview

This project involves building an interactive machine learning application to predict breast cancer using the Breast Cancer dataset from `sklearn`. The app is developed using Python and Streamlit, allowing users to input data and view predictions about whether the cancer is malignant or benign.

### Key Features:

- **Data Loading and Preprocessing:** The dataset is loaded, cleaned, and preprocessed for analysis.
- **Feature Selection:** Selects the most important features using `SelectKBest`.
- **Artificial Neural Network (ANN):** Implements a Multi-Layer Perceptron (MLP) classifier for prediction.
- **Interactive App:** A Streamlit app enables user interaction for prediction and visualization.
- **Model Optimization:** Hyperparameter tuning with Grid Search CV ensures optimal model performance.

---

## Requirements

To set up and run the project, ensure the following software and libraries are installed:

### Prerequisites:

1. Python 3.8 or higher
2. Virtual Environment (recommended)
3. Git

## Usage

### Step 1: Clone the Repository

Clone the GitHub repository to your local machine:

```bash
git clone <repository-url>
cd breast_cancer_analysis
```

### Step 2: Set Up Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Python Libraries:

Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit App

Run the Streamlit application locally:

```bash
streamlit run app.py
```

### Step 4: Interact with the App

Enter feature values in the input fields.
Click "Predict" to see the result (Malignant or Benign).
View the model's accuracy displayed in the app.
