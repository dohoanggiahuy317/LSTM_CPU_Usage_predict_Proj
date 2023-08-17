#------------------------------------------------------------------------------------
# IMPORT ALL NECESSARY MODULES
#------------------------------------------------------------------------------------

import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import argparse
import joblib
import tensorflow as tf
import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils import model_setup, callback_setup, np_array_convert


#------------------------------------------------------------------------------------
# Read the CSV file into DataFrame
#------------------------------------------------------------------------------------

def read_data(filename):
    df = pd.read_csv(filename)
    df = df.set_index('timestamp')
    return df


#------------------------------------------------------------------------------------
# Transform DataFrame into format expected by model for evaluation
#------------------------------------------------------------------------------------

def transform_data(df, prev_day, pred_day, scaler_path):

    # Chunk datafram into Features (number of days that model needs to know) and Labels (number of days model can predict)
    x_test, y_test = np_array_convert(df, prev = prev_day, pred = pred_day)


    # Normalize the data using the saved the parameters of normalization
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaler = joblib.load(scaler_path + '/scaler/scaler_features.pkl')
    x_test_scale = scaler.transform(pd.DataFrame(x_test))

    scaler = joblib.load(scaler_path + '/scaler/scaler_labels.pkl')
    y_test_scale = scaler.transform(pd.DataFrame(y_test))

    return x_test, y_test, x_test_scale, y_test_scale

#------------------------------------------------------------------------------------
# Print the test results
#------------------------------------------------------------------------------------

def test_score(preds, y_test, pred_day):

    # mean_squared_error
    testScore_1 = math.sqrt(mean_squared_error(y_test[:], preds[:]))
    print('Test Score: %.2f RMSE' % (testScore_1))

    # mean_absolute_error
    testScore_2 = math.sqrt(mean_absolute_error(y_test[:], preds[:]))
    print('Test Score: %f MAE' % (testScore_2))

    # MAPE
    testScore_3 = np.mean(np.abs(preds - y_test)/np.abs(y_test)*100)
    print('Test Score: %f MAPE' % (testScore_3))


    # mean absolute mean error
    arr = 1 - abs(y_test[:] - preds[:])/y_test[:]
    arr = arr = np.where(arr < 0, 0, arr)
    acc = sum(sum(arr)/len(y_test))/pred_day

    print("*** Accuracy: ", acc * 100)



#------------------------------------------------------------------------------------
# Main function
#------------------------------------------------------------------------------------


def main():

    # Get the parameters from the shell script that will be used for evaluation
    parser = argparse.ArgumentParser(description='Split the dataset into Dataframe training and testing')
    parser.add_argument('--eval_path',        type=str, help='Dataset File')
    parser.add_argument('--model_path',       type=str, help='Dataset File')
    parser.add_argument('--save_pred_path',   type=str, help='Dataset File')
    parser.add_argument('--prev_day',         type=int, help='Train test split ratio File',           default=144)
    parser.add_argument('--pred_day',         type=int, help='Window size for smoothing the dataset', default=12)
    args = parser.parse_args()

    # Read the dataset
    df = read_data(args.eval_path)

    # Normalize the dataset
    x_test, y_test, x_test_scale, y_test_scale = transform_data(df, args.prev_day, args.pred_day, args.model_path)


    # Setting up the model
    model = model_setup(x_test_scale, args.pred_day)
    model = tf.keras.models.load_model(args.model_path)

    # Predict
    preds = model.predict(x_test_scale)
    scaler = joblib.load(args.model_path + '/scaler/scaler_labels.pkl')
    preds = scaler.inverse_transform(preds)
    test_score(preds, y_test, args.pred_day)


    # Save the prediction
    # preds.to_csv(args.save_pred_path)
    print(preds)

main()