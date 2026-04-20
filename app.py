import streamlit as st
import pandas as pd
import numpy as np
import joblib
# from sklearn.base import BaseEstimator, TransformerMixin

# # --- PASTE THE CUSTOM CLASS HERE ---
# class InsulinPreprocessor(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self
        
#     def transform(self, X):
#         X_copy = X.copy() 
#         X_copy['Insulin'] = X_copy['Insulin'].replace(0, np.nan)
#         X_copy['Insulin_Missing'] = X_copy['Insulin'].isna().astype(int)
#         X_copy['Insulin'] = X_copy['Insulin'].fillna(-999)
#         return X_copy


# 1. Load your trained pipeline
# Make sure 'diabetes_production_model.pkl' is in the same folder as this script
@st.cache_resource
def load_model():
    return joblib.load('diabetes_production_model.pkl')

model = load_model()

# 2. Build the UI
st.title("🩺 Diabetes Prediction Engine")
st.write("Enter the patient's medical details below. The model will automatically handle missing values (like unknown Insulin) via the custom pipeline.")

st.divider()

# 3. Create input widgets for the features
# We set default values to make testing faster
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=110)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    
with col2:
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.000, max_value=3.000, value=0.400, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

# We highlight the Insulin field so the user knows they can leave it at 0
st.info("💡 **Note on Insulin:** If the insulin test wasn't performed, leave it as 0. Our pipeline will automatically convert it to a missing indicator and handle the math.")
insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, value=0)

st.divider()

# 4. Make the Prediction
if st.button("Predict Outcome", type="primary", use_container_width=True):
    
    # Pack the user inputs into a Pandas DataFrame exactly like the training data
    # (Notice we exclude 'SkinThickness' completely, just as our pipeline expects)
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, insulin, bmi, dpf, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # Feed it to the pipeline
    prediction = model.predict(input_data)
    
    # Display the result
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("⚠️ The model predicts a High Risk of Diabetes.")
    else:
        st.success("✅ The model predicts a Low Risk / Normal status.")