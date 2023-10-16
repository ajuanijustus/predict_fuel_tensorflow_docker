import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Load the saved model
model = keras.models.load_model('fuel_efficiency_model')

# Load test dataset
df = pd.read_csv('test.csv')

# Data preprocessing
df = df[df['horsepower'] != '?']
df['horsepower'] = df['horsepower'].astype(int)
df.drop('displacement', axis=1, inplace=True)

features = df.drop(['mpg', 'car name'], axis=1)
target = df['mpg'].values

# Sample input for prediction
sample_input = features.iloc[0].values.reshape(1, -1)  # Example input features for prediction
print("Actual MPG:", target[0])

# Perform prediction
predicted_mpg = model.predict(sample_input)

print("Predicted MPG:", predicted_mpg[0][0])
