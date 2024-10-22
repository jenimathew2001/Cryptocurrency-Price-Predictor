import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM , Dropout,LeakyReLU
from tensorflow.keras.metrics import MeanSquaredError
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler #scale data betn 0 to 1
import datetime as dt
from datetime import timezone
import pytz
from tensorflow import keras
import io
import random as rd
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime

data = pd.read_csv('coin_Bitcoin.csv')


"""# **DATA PREPROCESSING**

Adding a range column into the data
"""

y=[]
y = data['High'] - data['Low']
y = pd.DataFrame(y , columns = ["Range"])
data = pd.concat([data, y], axis=1)
data

import plotly.graph_objects as go

fig = go.Figure(data=go.Scatter(x=data['Date'], y=data['Range'], mode='lines'))
fig.update_layout(
    title='Cryptocurrency Range Trend',
    xaxis_title='Date',
    yaxis_title='Range',
    xaxis_rangeslider_visible=True
)
fig.show()

"""Visualize the data (Range vs date)

# **DATA SPLITTING**

NORMALIZING
"""

#Normalizing
range_1 = data[['Range']]
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()#this by default returns the value between 0-10.
norm_data= min_max_scaler.fit_transform(range_1.values)#fitting these values

"""SPLITTING THE DATA"""

# SPLITTING DATA

def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []
  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size
  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)

past_history = 5
future_target = 0
#we need to use 5 days of data to learn to predict the next point in the time series ‘future_target’.
TRAIN_SPLIT = int(len(norm_data) * 0.8)#Last index of the 80% data used for training.
x_train, y_train = univariate_data(norm_data,
                                   0,
                                   TRAIN_SPLIT,
                                   past_history,
                                   future_target)
x_test, y_test = univariate_data(norm_data,
                                 TRAIN_SPLIT,
                                 None,
                                 past_history,
                                 future_target)

"""# **BUILDING AND TRAINING THE MODEL**

BUILD THE MODEL
"""

# BUILD THE MODEL

num_units = 64#Number of neurons
learning_rate = 0.0001
activation_function = 'sigmoid'
adam = Adam(learning_rate=learning_rate)
loss_function = 'mse'
batch_size = 10
num_epochs = 1000
#NUMBER OF INPUT PARAMETERS
# for four imput parameter  N = 4 and for one input parameter
N = 1 #C

# Initialize the RNN
model = Sequential()
#In Keras we can simply stack multiple layers on top of each other, for this we need to initialize the model as Sequential().
model.add(LSTM(units = num_units, activation=activation_function, input_shape=(None, N)))
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.1))
#This layer will help to prevent overfitting by ignoring randomly selected neurons during training, and hence reduces the sensitivity to the specific weights of individual neurons.
model.add(Dense(units = 1))#fully conneceted layer

# Compiling the RNN
model.compile(optimizer=adam, loss=loss_function)

"""TRAINING THE MODEL"""

history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    batch_size=batch_size,
    epochs=num_epochs,
    shuffle=False
)

model.save('model.h5', save_format='h5')

"""# **MODEL EVALUATION**"""

original = pd.DataFrame(min_max_scaler.inverse_transform(y_test))
predictions = pd.DataFrame(min_max_scaler.inverse_transform(model.predict(x_test)))
sns.set(rc={'figure.figsize':(11.7+2,8.27+2)})
ax = sns.lineplot(x=original.index, y=original[0], label="Test Data", color='royalblue')
ax = sns.lineplot(x=predictions.index, y=predictions[0], label="Prediction", color='tomato')
ax.set_title('Bitcoin price', size = 14, fontweight='bold')
ax.set_xlabel("Days", size = 14)
ax.set_ylabel("Range", size = 14)
ax.set_xticklabels('', size=10)

#xgb_accuracy = round((accuracy_score(y_test, predictions) * 100), 2)
test_loss  = model.evaluate(x = x_test,y = y_test,verbose=1)
print('Test loss is : ',test_loss)