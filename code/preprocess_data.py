#-----------------------------------------------------------
# IMPORT ALL NECESSARY MODULES
#-----------------------------------------------------------
import os
import pandas as pd
import argparse


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


#---------------------------------------------------------------------------------------------------------------------
# CREATE TRAIN-TEST SPLIT (80:20)
#---------------------------------------------------------------------------------------------------------------------
def train_test_split(df, ratio):

    # Split the dataset into Dataframe training and testing with ratio
    train_length = round(len(df)*ratio)
    test_length = len(df) - train_length

    train = df.iloc[0: train_length]
    test = df[train_length :]

    return train, test



#-----------------------------------------------------------
# MAIN FUNCTION
#-----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Split the dataset into Dataframe training and testing')
    parser.add_argument('--dataset_path', type=str, help='Dataset File')
    parser.add_argument('--save_folder_path', type=str, help='Dataset File')
    parser.add_argument('--ratio', type=float, help='Train test split ratio File',    default=0.8)
    parser.add_argument('--group', type=str, help='Time step for model', default=None)
    args = parser.parse_args()


    # Dataset statistics
    print("---- Spliting the dataset -----")
    print("DATASET STATISTICS")
    df = read_data(args.dataset_path, args.group)
    train, test = train_test_split(df, args.ratio)

    print("Number of rows in the Train:", train.shape[0])
    print("Number of rows in the Test:", test.shape[0])

    # Split the dataset
    print("---- Split data successfully -----")
    print("Saving data...")

    if not os.path.exists(args.save_folder_path):
        os.makedirs(args.save_folder_path)

    train.to_csv(args.save_folder_path + "/train.csv")
    test.to_csv(args.save_folder_path + "/test.csv")

    print("Saved")

main()