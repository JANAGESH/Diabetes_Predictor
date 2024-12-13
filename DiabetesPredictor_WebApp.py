# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 23:00:35 2024

@author: janag
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('D:/Machine Learning Projects/Diabetes_Predictor/trained_model.sav', 'rb'))

#creating a function for prediction 

def diabetes_prediction(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0] == 1):
      return "The person is diabetic"
    else:
      return "The person is not diabetic"
  
def main():
    
    #Giving a title
    
    st.title('Diabetes Predictor App')
    
    #Getting the input from the user
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Thickness of Skin')
    Insulin = st.text_input('Insulin Value')
    BMI = st.text_input('BMI INDEX')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('AGE')
    
    #Code for the prediction
    
    diagnosis = ''
    
    #creating a button for the prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()