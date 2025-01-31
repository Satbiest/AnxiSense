import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
mental = pd.read_csv("C:/Users/resia/Downloads/Dataset-Mental-Disorders.csv")

# Preprocessing
yes_and_no_columns = ['Suicidal thoughts']
for column in yes_and_no_columns:
    mental[column] = mental[column].str.strip().str.upper().fillna("NO")
    mental[column] = mental[column].map({'YES': 1, 'NO': 0})

# Map expert diagnosis to integer values
mapping_dict = {'Normal': 0, 'Bipolar Type-1': 1, 'Bipolar Type-2': 2, 'Depression': 3}
mental['Expert Diagnose'] = mental['Expert Diagnose'].map(mapping_dict).astype(int)

# Drop non-feature columns
X = mental.drop(columns=['Expert Diagnose', 'Patient Number'])
y = mental['Expert Diagnose']

# Convert categorical columns to numerical
categorical_columns = X.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # Store encoders for later use

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)

# Streamlit UI
st.title("Anxiety Detection Model")
st.write("This model detects the type of mental disorder based on input responses.")

# Collect user input for prediction
user_input = {}
for column in X.columns:
    if column in categorical_columns:
        options = label_encoders[column].classes_
        user_input[column] = st.selectbox(f"Select {column}", options)
    elif column == 'Suicidal thoughts':
        user_input[column] = st.selectbox("Select Suicidal thoughts", ['No', 'Yes'])
        user_input[column] = 1 if user_input[column] == 'Yes' else 0
    else:
        user_input[column] = st.slider(f"Select value for {column}", int(X[column].min()), int(X[column].max()))

# Convert user input to the model input format
for col in categorical_columns:
    user_input[col] = label_encoders[col].transform([user_input[col]])[0]

user_input_df = pd.DataFrame([user_input])

# Make prediction using the trained model
prediction = xgb_classifier.predict(user_input_df)
prediction_proba = xgb_classifier.predict_proba(user_input_df)

# Show the prediction result
diagnosis_dict = {0: 'Normal', 1: 'Bipolar Type-1', 2: 'Bipolar Type-2', 3: 'Depression'}
st.write(f"Predicted Diagnosis: {diagnosis_dict[prediction[0]]}")

# Display probability of each class
st.write("Probability of each disorder:")
for idx, disorder in diagnosis_dict.items():
    st.write(f"{disorder}: {prediction_proba[0][idx]*100:.2f}%")