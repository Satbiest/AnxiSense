import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# === Caching Dataset ===
@st.cache_data
def load_data():
    return pd.read_csv('Dataset-Mental-Disorders.csv')

mental = load_data()

# === Preprocessing ===
# Convert Yes/No columns
mental['Suicidal thoughts'] = mental['Suicidal thoughts'].str.strip().str.upper().fillna("NO").replace({'YES': 1, 'NO': 0})

# Map expert diagnosis
mapping_dict = {'Normal': 0, 'Bipolar Type-1': 1, 'Bipolar Type-2': 2, 'Depression': 3}
mental['Expert Diagnose'] = mental['Expert Diagnose'].map(mapping_dict).astype(int)

# Drop non-feature columns
X = mental.drop(columns=['Expert Diagnose', 'Patient Number'])
y = mental['Expert Diagnose']

# Convert categorical columns to numerical
categorical_columns = X.select_dtypes(include=['object']).columns
label_encoders = {col: LabelEncoder().fit(X[col]) for col in categorical_columns}

for col in categorical_columns:
    X[col] = label_encoders[col].transform(X[col])

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Load or Train Model ===
MODEL_PATH = "xgb_model.pkl"

try:
    xgb_classifier = joblib.load(MODEL_PATH)
except FileNotFoundError:
    xgb_classifier = XGBClassifier()
    xgb_classifier.fit(X_train, y_train)
    joblib.dump(xgb_classifier, MODEL_PATH)

# === Feature Importance ===
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_classifier.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Select top 10 most important features
top_features = feature_importance_df['Feature'][:10].tolist()

# === Streamlit UI ===
st.title("AnxiSense: Check Your Tension, Know Your Emotion")
st.write("This model detects the type of mental disorder based on input responses.")

# Sidebar for user input
st.sidebar.header("User Input")
user_input = {}

for column in top_features:
    if column in categorical_columns:
        options = label_encoders[column].classes_
        user_input[column] = st.sidebar.selectbox(f"{column}", options)
    elif column == 'Suicidal thoughts':
        user_input[column] = st.sidebar.selectbox("Suicidal thoughts", ['No', 'Yes'])
        user_input[column] = 1 if user_input[column] == 'Yes' else 0
    else:
        user_input[column] = st.sidebar.slider(f"{column}", int(X[column].min()), int(X[column].max()))

# Convert user input to DataFrame
for col in categorical_columns:
    if col in top_features:
        user_input[col] = label_encoders[col].transform([user_input[col]])[0]

user_input_df = pd.DataFrame([user_input])

# === Prediction Button ===
if st.sidebar.button("Check your condition"):
    prediction = xgb_classifier.predict(user_input_df)
    prediction_proba = xgb_classifier.predict_proba(user_input_df)

    diagnosis_dict = {0: 'Normal', 1: 'Bipolar Type-1', 2: 'Bipolar Type-2', 3: 'Depression'}
    predicted_diagnosis = diagnosis_dict[prediction[0]]

    # Color the diagnosis based on the prediction
    if predicted_diagnosis == 'Normal':
        st.markdown(f"### Predicted Diagnosis: **<span style='color:green'>{predicted_diagnosis}</span>**", unsafe_allow_html=True)
    else:
        st.markdown(f"### Predicted Diagnosis: **<span style='color:red'>{predicted_diagnosis}</span>**", unsafe_allow_html=True)

    st.write("### Probability of each disorder:")
    for idx, disorder in diagnosis_dict.items():
        st.write(f"{disorder}: **{prediction_proba[0][idx]*100:.2f}%**")
