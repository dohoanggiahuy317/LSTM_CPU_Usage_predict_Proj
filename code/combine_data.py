#-----------------------------------------------------------
# IMPORT ALL NECESSARY MODULES
#-----------------------------------------------------------
import os

import pandas as pd
import argparse

def main():

    parser = argparse.ArgumentParser(description='Split the dataset into Dataframe training and testing')
    parser.add_argument('--dataset_folder',     type=str, help='Dataset File')
    parser.add_argument('--save_folder',        type=str, help='Dataset File')
    args = parser.parse_args()

    csv_files = [file for file in os.listdir(args.dataset_folder) if file.endswith(".csv")]
    dataframes_dict = {}

    for csv_file in csv_files:
        csv_file_path = os.path.join(args.dataset_folder, csv_file)
        df = pd.read_csv(csv_file_path)
        hostname = df['hostname'][0]  # Get the hostname from the first row
        dataframes_dict[hostname] = df

    # Find the common hostnames in all dataframes
    common_hostnames = set(dataframes_dict.keys())
    for df in dataframes_dict.values():
        common_hostnames = common_hostnames.intersection(df['hostname'])

    # Initialize an empty list to store the dataframes after filtering
    filtered_dataframes_list = []

    # Filter dataframes to include only common hostnames
    for hostname in common_hostnames:
        filtered_dataframes_list.append(dataframes_dict[hostname])

    # Combine filtered dataframes into a single dataframe
    combined_filtered_df = pd.concat(filtered_dataframes_list, ignore_index=True)
    combined_filtered_df['timestamp'] = pd.to_datetime(combined_filtered_df['timestamp'])
    combined_df = combined_filtered_df.set_index('hostname')

    combined_df.to_csv(args.save_folder + "/combine.csv")
    print("Data saved to " + args.save_folder + "/combine.csv")

main()