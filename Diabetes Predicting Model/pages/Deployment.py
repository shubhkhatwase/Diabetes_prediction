import streamlit as st
import pandas as pd
import pickle
st.set_page_config(page_title="Deployment", page_icon="📈")

model=pickle.load(open('diabetes_model.sav','rb'))

# prediction =None



def head(url):
     st.markdown(f'<p style="background-color:#f4c2c2 ;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)

head("(人◕‿◕) 𝔼𝕟𝕥𝕖𝕣 𝕧𝕒𝕝𝕦𝕖𝕤 𝕥𝕠 𝕔𝕙𝕖𝕔𝕜 (•◡•))")

col1, col2, col3 = st.columns(3)
    
with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
with col2:
        Glucose = st.text_input('Glucose Level')
    
with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
with col2:
        Insulin = st.text_input('Insulin Level')
    
with col3:
        BMI = st.text_input('BMI value')
    
with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
diab_diagnosis = ''
    
    # creating a button for Prediction
    
if st.button('Diabetes Test Result'):
        diab_prediction = model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
st.success(diab_diagnosis)