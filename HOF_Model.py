'''
Created on Dec 24, 2018

@author: togunyale
'''
import tensorflow as tf
import pandas as pd 
import numpy as np
from pandas.io.parsers import read_csv
from tensorflow.python.data import Dataset
import math
from sklearn import metrics


def main():
    learningRate = 0.003
    steps = 10
    Epochs = 200000
    periods = 10
    batch_size = 500
    stepsPerPeriod = steps/periods
    
    tf.logging.set_verbosity(tf.logging.ERROR) 
    pd.options.display.max_rows = 10
    pd.options.display.float_format = '{:.3f}'.format
    
    raw_data = read_csv("train.csv", sep=",")
    data = raw_data.copy()
    data = data.reindex(np.random.permutation(data.index))
    
    
    "G : 0 , F : 1 , C:2 , F-G:3 , F-C:4 -> New Values for Positions"
    cleanUp = {"Pos": {'G': 0 , 'PG': 0 , 'SG': 0 , 'F': 1 , 'PF': 1 , 'SF': 1 , 'C': 2, 'C-F':4 , 'F-G' : 3, 'F - G': 3 , 'G-F':3, 'C - F':4, 'F - C ':4 , 'F-C' : 4 } }
    data.replace(cleanUp, inplace=True)
    
    "Synthetic Features in perspective of per game (PG) stats"
    data["FG"] = (data['2PM'] + data['3PM']) / (data['2PA'] + data['3PA'])
    data["PPG"] = data['PTS'] / data['GP']
    data["ASTPG"] = data['AST'] / data['GP']
    data["RBPG"] = data['TRB'] / data['GP']
    data["STLPG"] = data['STL'] / data['GP']
    data["BLKPG"] = data['BLK'] / data['GP']
    
    
    train = data.sample(frac=0.8,random_state=0)
    test = data.drop(train.index)
    
    trainF = train[[ "Ht", "Wt", "Pos", "GP", "MP", "RY", "LY", "EXP", "PTS", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "FG" , "PPG", "ASTPG", "RBPG", "STLPG", "BLKPG"]]
    testF = test[[ "Ht", "Wt", "Pos", "GP", "MP", "RY", "LY", "EXP", "PTS", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "FG" , "PPG", "ASTPG", "RBPG", "STLPG", "BLKPG"]]
    
    
    
    trainL = train["HOF"]
    testL = test["HOF"]
    
    print "Training Features : " 
    print trainF.describe().to_string()
    print " "
    print "Training Labels : "
    print trainL.describe().to_string()
    print " "
    print "Test features : "
    print testF.describe().to_string()
    print " "
    print "Test Labels : "
    print testL.describe().to_string()
    print " "
    
    
    
    
    
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
    feature_columns= set([tf.feature_column.numeric_column(my_feature) for my_feature in trainF]),
    optimizer=my_optimizer
    )
    
    train_func = lambda: input_Func(trainF, trainL, batch_size=batch_size)
    train_predict_func = lambda: input_Func(trainF, trainL, num_epochs=1 , shuffle=False)
    test_predict_func = lambda: input_Func(testF, testL, num_epochs=1 , shuffle=False)
    
    print "Training Model ..."
    print "RMSE on Training Data : "
    train_rmse = []
    test_rmse = []
    for period in range(0 , periods):
        print "Here : STEP 1 "
        linear_regressor.train(
            input_fn=train_func,
            steps=stepsPerPeriod
        )
        
        # Take a break and compute predictions.
        training_predictions = linear_regressor.predict(input_fn=train_predict_func)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        
        validation_predictions = linear_regressor.predict(input_fn=test_predict_func)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
        print "Here : STEP 2 "
        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, trainL))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, testL))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        train_rmse.append(training_root_mean_squared_error)
        test_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")


def input_Func(features , targets , batch_size = 1 , shuffle = True , num_epochs=None):
    features = {key:np.array(value) for key,value in dict(features).items()}                                            
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

if __name__== "__main__":
    main()
    
    

