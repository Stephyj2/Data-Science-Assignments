# app.py
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')

# Function to make predictions
def predict_survival(features):
    prediction = model.predict([features])
    return prediction[0]

# Streamlit app
st.title('Titanic Survival Prediction')

# Input features from user
pclass = st.selectbox('Pclass', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 100, 25)
sibsp = st.number_input('SibSp', 0, 10, 0)
parch = st.number_input('Parch', 0, 10, 0)
fare = st.number_input('Fare', 0.0, 520.0, 32.0)
#cabin_available = st.selectbox('Cabin Available', [0, 1])
embarked_q = st.selectbox('Embarked_Q', [0, 1])
embarked_s = st.selectbox('Embarked_S', [0, 1])

# Convert sex to numerical value
sex_male = 1 if sex == 'male' else 0

# Create a feature array
features = [pclass, age, sibsp, parch, fare, embarked_q, embarked_s, sex_male]

# Predict survival
if st.button('Predict'):
    result = predict_survival(features)
    if result == 1:
        st.success('The passenger would have survived.')
    else:
        st.error('The passenger would not have survived.')
