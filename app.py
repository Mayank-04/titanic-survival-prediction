import streamlit as st
import pandas as pd
import pickle

# ✅ Load the trained model once at the top
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Titanic Survival Prediction 🚢")
st.title("🎯 Titanic Survival Prediction App")
st.markdown("Enter passenger details to predict survival.")

# 💡 Input form
with st.form("input_form"):
    Pclass = st.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
    Sex_input = st.selectbox("Sex", ["male", "female"])
    Age = st.slider("Age", 0, 100, 25)
    SibSp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    Parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
    Fare = st.number_input("Fare", min_value=0.0, value=10.0, format="%.2f")
    Embarked_input = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
    submit = st.form_submit_button("Predict")

if submit:
    # 🔄 Encode categorical values
    Sex = 1 if Sex_input == "male" else 0
    Embarked_map = {"C": 0, "Q": 1, "S": 2}
    Embarked = Embarked_map[Embarked_input]

    # 🚫 DO NOT overwrite 'model' variable here!
    input_data = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]],
                              columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])

    # ✅ Use model to predict
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.success(f"🎉 Prediction: Survived (Confidence: {proba*100:.2f}%)")
    else:
        st.error(f"💀 Prediction: Did NOT Survive (Confidence: {(1-proba)*100:.2f}%)")
