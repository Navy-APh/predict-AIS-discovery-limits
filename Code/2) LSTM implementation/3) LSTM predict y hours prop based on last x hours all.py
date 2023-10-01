no_of_rings = 10

no_recorded_x_hours = 72
no_predicted_y_hours = 24

# Importing the libraries
import numpy as np
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/Colab Notebooks')

# Importing the dataset
dataset1_train = pd.read_csv("Period 1_sanitized_and_processed.csv")
dataset2_test  = pd.read_csv("Period 2_sanitized_and_processed.csv")

#_______________________________________________________________________________

len1 = len(dataset1_train)
len2 = len(dataset2_test)

# Merging the datasets
dataset = pd.concat([dataset1_train, dataset2_test], ignore_index=True)

X = dataset.iloc[:, 0:10].values
Y = dataset.iloc[:, 10:(10 + 1 * no_of_rings)].values

# Feature Scaling on first 12 items
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

ALL = np.concatenate((X, Y), axis=1)

dataset1_train = ALL[:len1, :]
dataset2_test  = ALL[len1:, :]

#____________________________________________________________________________

rec = no_recorded_x_hours
pred = no_predicted_y_hours

X_train = []
y_train = []
for i in range(len(dataset1_train) - rec - pred + 1):
    X_train.append(dataset1_train[i:i+rec, 1:10].reshape(1, -1))
    y_train.append(dataset1_train[i+rec:i+rec+pred, 10:(10 + 1 * no_of_rings)].reshape(1, -1))
# Converting X_train and y_train to NumPy arrays
X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

# Initializing X_test and y_test as empty lists
X_test = []
y_test = []
for i in range(len(dataset2_test) - rec - pred + 1):
    X_test.append(dataset2_test[i:i+rec, 1:10].reshape(1, -1))
    y_test.append(dataset2_test[i+rec:i+rec+pred, 10:(10 + 1 * no_of_rings)].reshape(1, -1))
# Converting X_test and y_test to NumPy arrays
X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)

input_shape = (X_train.shape[1], 1)

input_neurons = X_train.shape[1]
output_neurons = y_train.shape[1]

X_train = np.reshape(X_train, (X_train.shape[0], *input_shape))
X_test = np.reshape(X_test, (X_test.shape[0], *input_shape))

print(input_neurons)
print(output_neurons)

# Define the learning rate scheduler function
def lr_scheduler(epoch, lr):
    lr = 0.01
    if epoch > 20:
        lr = lr / 10
    if epoch > 40:
        lr = lr / 10
    if epoch > 60:
        lr = lr / 10
    if epoch > 80:
        lr = lr / 10
    return lr

# Define the learning rate scheduler function
from tensorflow.keras.callbacks import LearningRateScheduler

lr_callback = LearningRateScheduler(lr_scheduler)

# Define the learning rate scheduler function
from tensorflow.keras.callbacks import LearningRateScheduler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.metrics import Precision
from tensorflow.keras.layers import LSTM

# Defining the neural network model
model = Sequential()

model.add(LSTM(1024, input_shape = (input_neurons,1), return_sequences=True))
model.add(LSTM(units=1024))
for i in range (1,4):
  model.add(Dense(2048, activation=LeakyReLU(alpha=0.3)))

model.add(Dense(output_neurons, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['binary_accuracy', Precision()])

# Training the model
model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test), shuffle=False)
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), callbacks=[lr_callback], shuffle=False)

print()
print("Generating datasets")
print()

# Predicting on X
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Creating a DataFrame of X and y_pred
result_test_df = pd.DataFrame(np.concatenate((y_test, y_pred_test), axis=1))
result_train_df = pd.DataFrame(np.concatenate((y_train, y_pred_train), axis=1))

# Specify the path to save the CSV file on Google Drive
save_path_test =  '/content/drive/MyDrive/Colab Notebooks/Results/LSTM_All_X_test_results_comparison.csv'
save_path_train = '/content/drive/MyDrive/Colab Notebooks/Results/LSTM_All_X_train_results_comparison.csv'

# Save the DataFrame as a CSV file
result_test_df.to_csv(save_path_test, index=False)
result_train_df.to_csv(save_path_train, index=False)

print()
print("All done!")
print()



