import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Load train dataset
df = pd.read_csv('train.csv')

# Data preprocessing
df = df[df['horsepower'] != '?']
df['horsepower'] = df['horsepower'].astype(int)
df.drop('displacement',
        axis=1,
        inplace=True)

features = df.drop(['mpg', 'car name'], axis=1)
target = df['mpg'].values

X_train, X_val, Y_train, Y_val = train_test_split(features, target,
                                      test_size=0.2,
                                      random_state=22)

# Data Input Pipeline
AUTO = tf.data.experimental.AUTOTUNE

train_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_train, Y_train))
    .batch(32)
    .prefetch(AUTO)
)

val_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_val, Y_val))
    .batch(32)
    .prefetch(AUTO)
)

# Model Architecture
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=[6]),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1, activation='relu')
])

# Compile the model
model.compile(
    loss='mae',
    optimizer='adam',
    metrics=['mape']
)

# Model Training
history = model.fit(train_ds, epochs=50, validation_data=val_ds)

# Save the trained model
model.save('fuel_efficiency_model')
