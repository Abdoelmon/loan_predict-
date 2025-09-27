import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# Load the model
model = joblib.load('model.pkl')
st.title('Loan Prediction Model')
Gender = st.selectbox('Gender', options=['Male', 'Female'])
Married = st.selectbox('Married', options=['Yes', 'No'])
Dependents = st.selectbox('Dependents', options=['0', '1', '2', '3+'])
Education = st.selectbox('Education', options=['Graduate', 'Not Graduate'])
Self_Employed = st.selectbox('Self Employed', options=['Yes', 'No'])
Applicant_Income = st.number_input('Applicant Income', min_value=0, value=5000)
Coapplicant_Income = st.number_input('Coapplicant Income', min_value=0, value=2000)
Loan_Amount = st.number_input('Loan Amount', min_value=0, value=100000)
Loan_Amount_Term = st.number_input('Loan Amount Term', min_value=0, value=360)
Credit_History = st.number_input('Credit History (1 or 0)', min_value=0, max_value=1, value=1)
Property_Area = st.selectbox('Property Area', options=['Urban', 'Semiurban', 'Rural'])

input_data = pd.DataFrame({
    'Gender': [Gender],
    'Married': [Married],
    'Dependents': [Dependents],
    'Education': [Education],
    'Self_Employed': [Self_Employed],
    'Applicant_Income': [Applicant_Income],
    'Coapplicant_Income': [Coapplicant_Income],
    'Loan_Amount': [Loan_Amount],
    'Loan_Amount_Term': [Loan_Amount_Term],
    'Credit_History': [Credit_History],
    'Property_Area': [Property_Area]
})
# Preprocess input data
input_data.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

# replace 3+ with 3
input_data['Dependents'].replace('3+', 3, inplace=True)
input_data['Dependents'] = input_data['Dependents'].astype(float)

input_data.Applicant_Income = np.sqrt(input_data.Applicant_Income)
input_data.Coapplicant_Income = np.sqrt(input_data.Coapplicant_Income)
input_data.Loan_Amount = np.sqrt(input_data.Loan_Amount)

scaler = MinMaxScaler()
input_data = scaler.fit_transform(input_data)

if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success('Loan Approved')
    else:
        st.error('Loan Not Approved')
        