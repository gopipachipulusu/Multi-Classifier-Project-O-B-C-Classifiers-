# -*- coding: utf-8 -*-
############ Please Make sure to update the datasets folder next to the folder ############

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


# A combined dataframe has been populated and the logistic regression model has been used to classify all 'C' type dataset values.

folder = 'data' 
directory = os.getcwd()
path = os.path.join(directory, folder)

csv_files = [file for file in os.listdir(path) if file.endswith('.csv')]

dataframes = []

for file in csv_files:
    file_path = os.path.join(path, file)
    df = pd.read_csv(file_path, header=None, names=['Label', 'Values'])
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = combined_df['Values'].values.reshape(-1, 1)
y = combined_df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

#saving the trained model as a pickle file
import pickle


def classify_c(input_vector):
#retriving the model from pickle file
    file_path='Group_7_Week_8.pkl' #pickle file path
    with open(file_path, 'wb') as file:
        pickle.dump(log_model,file)
        
    output_vector = []
    # Reshape the input vector to a 2D array
    input_vector_2d = np.array(input_vector).reshape(-1, 1)
    # Predict using the logistic regression model
    predictions = log_model.predict(input_vector_2d)
    for prediction in predictions:
        output_vector.append(prediction)
    return output_vector

input_vector = [78.57219088, 235.9270539, 229.6014935, 227.3320426, 230.7789018, 57.9394171, 59.47807172, 229.722738, 132]
output = classify_c(input_vector)
print("Classified labels:", output)

