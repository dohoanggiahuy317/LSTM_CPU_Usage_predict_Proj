#------------------------------------------------------------------------------------
# IMPORT ALL NECESSARY MODULES
#------------------------------------------------------------------------------------
import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import argparse
import joblib

from utils import model_setup, callback_setup, np_array_convert



#------------------------------------------------------------------------------------
# Read the CSV file into DataFrame
#------------------------------------------------------------------------------------

def read_data(filename):
    df = pd.read_csv(filename)
    df = df.set_index('timestamp')
    return df


#------------------------------------------------------------------------------------
# Transform DataFrame into format expected by model for training
#------------------------------------------------------------------------------------
def transform_data(df, prev_day, pred_day, model_path):

    # Chunk datafram into Features (number of days that model needs to know) and Labels (number of days model can predict)
    x_train, y_train = np_array_convert(df, prev = prev_day, pred = pred_day)

    # Scaler initialization
    scaler = MinMaxScaler(feature_range=(0, 1))
    if not os.path.exists(model_path + "/scaler"):
        os.makedirs(model_path + "/scaler")

    # Normalize the data and save the parameters of normalization
    x_train_scale = scaler.fit_transform(pd.DataFrame(x_train))
    joblib.dump(scaler, model_path + '/scaler/scaler_features.pkl')

    y_train_scale = scaler.fit_transform(pd.DataFrame(y_train))
    joblib.dump(scaler, model_path + '/scaler/scaler_labels.pkl')

    return x_train, y_train, x_train_scale, y_train_scale



#------------------------------------------------------------------------------------
# Main function
#------------------------------------------------------------------------------------

def main():

    # Get the parameters from the shell script that will be used for training
    parser = argparse.ArgumentParser(description='Training model')
    parser.add_argument('--train_path', type=str, help='Dataset File')
    parser.add_argument('--save_model_path', type=str, help='Dataset File')
    parser.add_argument('--window', type=int, help='Window size for smoothing the dataset', default=10)
    parser.add_argument('--prev_day', type=int, help='Number of days that model needs to know', default=144)
    parser.add_argument('--pred_day', type=int, help='Number of days model can predict', default=12)
    parser.add_argument('--epochs', type=int, help='Number of training epochs', default=5)
    parser.add_argument('--batch_size', type=int, help='Number of training batchsize (64, 128, 256)', default=128)
    parser.add_argument('--verbose', type=int, help='Output during training',default=1)
    parser.add_argument('--shuffle', type=bool,help='Shuffle the training dataset', default=False)
    args = parser.parse_args()


    # Read the dataset
    df = read_data(args.train_path)

    # Normalize the dataset
    train_df = df.rolling(window=10).mean().dropna()
    _, _, x_train_scale, y_train_scale = transform_data(train_df, args.prev_day, args.pred_day, args.save_model_path)


    # Setting up the model
    model = model_setup(x_train_scale, args.pred_day)
    callbacks = callback_setup()


    # Training
    model.fit(  x_train_scale, 
                y_train_scale,
                epochs      =args.epochs,
                batch_size  =args.batch_size,
                verbose     =args.verbose, 
                shuffle     =args.shuffle, 
                callbacks   =callbacks
            )

    subfolder_path = "/".join(args.save_model_path.split("/")[:-1])
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Save the model
    model.save(args.save_model_path, save_format='tf')

main()