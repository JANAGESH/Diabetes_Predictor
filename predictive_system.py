# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#loading the saved model

loaded_model = pickle.load(open('D:/Machine Learning Projects/Diabetes_Predictor/trained_model.sav', 'rb'))

input_data = (11,143,94,33,146,36.6,0.254,51)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the input data


prediction = loaded_model.predict(input_data_reshaped)

if (prediction[0] == 1):
  print("The person is diabetic")
else:
  print("The person is not diabetic")