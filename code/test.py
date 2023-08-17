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


#---------------------------------------------------------------------------------------------------------------------
# LOAD THE DATASET AND PLOT THE OBSERVATIONS
#---------------------------------------------------------------------------------------------------------------------
def read_data(filename, group_data = None):
    raw_df = pd.read_csv(filename)
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])

    if group_data is not None:
        raw_df.set_index('timestamp', inplace=True)
        raw_df = raw_df.groupby('hostname').resample(group_data).mean().reset_index()

    # Use pivot to transform the DataFrame
    raw_df = raw_df.pivot_table(index='timestamp', columns='hostname', values='avg')

    # Set the timestamp as index
    raw_df.reset_index(inplace=True)
    raw_df.columns.name = None
    raw_df = raw_df.set_index('timestamp')

    # Add missing frequency values
    frequency = '5T' if group_data is None else group_data
    idx = pd.date_range(start=raw_df.index.min(), end=raw_df.index.max(), freq=frequency)
    full_time_series_df = raw_df.reindex(idx)
    full_time_series_df.index.name = 'timestamp'

    # Handle empty data row
    full_time_series_df.isna().sum()
    df = full_time_series_df.interpolate(method='linear')

    return df


#------------------------------------------------------------------------------------
# Transform DataFrame into format expected by model for evaluation
#------------------------------------------------------------------------------------
def transform_data(df, prev_day, scaler_path):

    # Chunk datafram into Features (number of days that model needs to know) and Labels (number of days model can predict)
    x_test, y_test = np_array_convert(df, prev = prev_day, pred = 0)


    # Normalize the data using the saved the parameters of normalization
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaler = joblib.load(scaler_path + '/scaler/scaler_features.pkl')
    x_test_scale = scaler.transform(pd.DataFrame(x_test))

    # scaler = joblib.load(scaler_path + '/scaler/scaler_labels.pkl')
    # y_test_scale = scaler.transform(pd.DataFrame(y_test))

    return x_test, x_test_scale


#------------------------------------------------------------------------------------
# Main function
#------------------------------------------------------------------------------------
def main():

    # Get the parameters from the shell script that will be used for evaluation
    parser = argparse.ArgumentParser(description='Split the dataset into Dataframe training and testing')
    parser.add_argument('--eval_path', type=str, help='Dataset File')
    parser.add_argument('--model_path', type=str, help='Dataset File')
    parser.add_argument('--save_pred_path', type=str, help='Dataset File')
    parser.add_argument('--prev_day', type=int, help='Train test split ratio File', default=144)
    parser.add_argument('--pred_day', type=int, help='Window size for smoothing the dataset', default=12)
    parser.add_argument('--group', type=str, help='group data info',    default=None)
    args = parser.parse_args()

    # Read the data
    df = read_data(args.eval_path, args.group)
    df = df.tail(args.prev_day)
    col = df.columns

    # Normalize the dataset
    x_test, x_test_scale = transform_data(df, args.prev_day, args.model_path)

    # Setting up the model
    model = model_setup(x_test_scale, args.pred_day)
    model = tf.keras.models.load_model(args.model_path)

    # Predict
    preds = model.predict(x_test_scale)
    scaler = joblib.load(args.model_path + '/scaler/scaler_labels.pkl')
    preds = scaler.inverse_transform(preds)

    # Save the prediction
    subfolder_path = "/".join(args.save_pred_path.split("/")[:-1])
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    preds_df = pd.DataFrame(preds.T, columns=[f"column_{i+1}" for i in range(preds.shape[0])])
    preds_df.columns = col
    preds_df.to_csv(args.save_pred_path)

main()