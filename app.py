import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("🚢 Titanic Survival Predictor")

# User Inputs
Pclass = st.selectbox("Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 1, 80)
SibSp = st.slider("Siblings/Spouse", 0, 5)
Parch = st.slider("Parents/Children", 0, 5)
Fare = st.slider("Fare", 0, 500)
Embarked = st.selectbox("Embarked", ['C', 'Q', 'S'])

# Encode inputs
Sex = 0 if Sex == 'male' else 1
Embarked = {'C': 0, 'Q': 1, 'S': 2}[Embarked]

features = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]],
                        columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])

# Predict
if st.button("Predict"):
    prediction = model.predict(features)
    st.success("Survived 🚢" if prediction[0] == 1 else "Did not survive 💀")