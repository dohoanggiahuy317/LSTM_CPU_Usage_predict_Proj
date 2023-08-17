#-----------------------------------------------------------
# IMPORT ALL NECESSARY MODULES
#-----------------------------------------------------------
import os

import pandas as pd
import argparse


#-----------------------------------------------------------
# MAIN FUNCTION
#-----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Combine all dataset')
    parser.add_argument('--dataset_folder', type=str, help='Dataset File')
    parser.add_argument('--save_folder', type=str, help='Dataset File')
    args = parser.parse_args()

    # Retrieve all the .csv file in the dataset folder
    csv_files = [file for file in os.listdir(args.dataset_folder) if file.endswith(".csv")]
    combined_df = pd.DataFrame()

    for csv_file in csv_files:
        csv_file_path = os.path.join(args.dataset_folder, csv_file)
        df = pd.read_csv(csv_file_path)
        combined_df = pd.concat([combined_df, df])


    # Remove all the duplicate timestamps and hostname
    combined_df = combined_df.set_index(['hostname', 'timestamp'])
    combined_df.to_csv(args.save_folder + "/combine.csv")
    print(str(combined_df.shape[0]) +  " data entries are saved to " + args.save_folder + "/combine.csv")

main()