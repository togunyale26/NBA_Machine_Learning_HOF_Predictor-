'''
Created on Dec 19, 2018

@author: togunyale
'''
import math
import os

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from PrintDot import PrintDot
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd 
from pandas import compat
from sklearn import metrics
import tensorflow as tf
from sklearn.tree import DecisionTreeRegressor
from tensorflow.python.data import Dataset
from pandas.io.parsers import read_csv


def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [HOF]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label='Val Error')
  plt.legend()
  plt.ylim([0, 5])
  plt.show()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$HOF^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label='Val Error')
  plt.legend()
  plt.ylim([0, 20])
  plt.show()


tf.logging.set_verbosity(tf.logging.ERROR) 
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.3f}'.format


raw_dataset = read_csv("train.csv", sep=",")
dataset = raw_dataset.copy()

print dataset.tail().to_string()



dataset = dataset.reindex(np.random.permutation(dataset.index))
"G : 0 , F : 1 , C:2 , F-G:3 , F-C:4 "
cleanUp = {"Pos": {'G': 0 , 'PG': 0 , 'SG': 0 , 'F': 1 , 'PF': 1 , 'SF': 1 , 'C': 2, 'C-F':4 , 'F-G' : 3, 'F - G': 3 , 'G-F':3, 'C - F':4, 'F - C ':4 , 'F-C' : 4 } }
dataset.replace(cleanUp, inplace=True)

dataset["FG%"] = (dataset['2PM'] + dataset['3PM']) / (dataset['2PA'] + dataset['3PA'])
dataset["PPG"] = dataset['PTS'] / dataset['GP']
dataset["ASTPG"] = dataset['AST'] / dataset['GP']
dataset["RBPG"] = dataset['TRB'] / dataset['GP']
dataset["STLPG"] = dataset['STL'] / dataset['GP']
dataset["BLKPG"] = dataset['BLK'] / dataset['GP']
#dataset["Sum"] = dataset["PTS"] + dataset["ORB"] + data["DRB"] + data["TRB"] + data["AST"] + data["STL"] + data["BLK"] + data["GP"] + data["MP"] + data["EXP"]
dataset = dataset.drop(columns=['Player'])

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#sns.pairplot(train_dataset[["HOF", "Ht", "Wt", "Pos", "PTS", "TRB", "AST", "STL", "BLK"]], diag_kind="kde")
#plt.show()

print " "
train_stats = train_dataset.describe()

train_stats = train_stats.transpose()
print train_stats.describe().to_string()
print " "

train_labels = train_dataset.pop('HOF')
test_labels = test_dataset.pop('HOF')

print train_labels.head().to_string()

normed_train_data = (train_dataset - train_stats['mean']) / train_stats['std']
normed_test_data = (test_dataset - train_stats['mean']) / train_stats['std']

model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

optimizer = tf.train.RMSPropOptimizer(0.001)

model.compile(loss='mse',
     optimizer=optimizer,
     metrics=['mae', 'mse'])

print model.summary()

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)

print example_result

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

print hist.tail().to_string()

plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} HOF".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

plt.show()

