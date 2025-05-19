import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Titanic Survival Predictor 🚢")
st.title("🎯 Titanic Survival Prediction App")
st.markdown("Enter passenger details to predict survival.")

# --- Input form ---
with st.form("input_form"):
    Pclass = st.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
    Sex = st.selectbox("Sex", ["male", "female"])
    Age = st.slider("Age", 0, 100, 25)
    SibSp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    Parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
    Fare = st.number_input("Fare Paid", min_value=0.0, value=10.0, format="%.2f")
    Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])  # C = Cherbourg, Q = Queenstown, S = Southampton

    submit = st.form_submit_button("Predict")

# --- Preprocessing ---
if submit:
    # Map categorical features
    Sex = 1 if Sex == "male" else 0
    Embarked_map = {"C": 0, "Q": 1, "S": 2}
    Embarked = Embarked_map[Embarked]

    # Construct DataFrame
    features = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]],
                            columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])
    
    # Model Prediction
    prediction = model.predict(features)
    proba = model.predict_proba(features)[0][1]

    # Display result
    if prediction[0] == 1:
        st.success(f"🎉 Prediction: Survived with {proba*100:.2f}% confidence.")
    else:
        st.error(f"💀 Prediction: Did NOT survive (Survival probability: {proba*100:.2f}%)")