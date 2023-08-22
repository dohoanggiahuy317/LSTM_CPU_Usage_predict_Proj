# 1. Setting up the Python Environment
1. All the necessary packages are available in the `requirements.txt` file.
    ```
    pip install -r requirements.txt
    ```

2. Activate the GPU environment for the TensorFlow library using the following shell script:
    ```
    env.sh
    ```

# 2. Steps for Training Models
To train two models for predicting RAM and CPU performance, follow these steps:

1. **Saving the Data** <br>
Place the data into two separate folders named CPU and RAM. <br>
The current training data spans from August 1, 2023, to August 21, 2023. It's important to save data of the same type for consecutive days in these folders.
    ```
    /data/original_data
    ```

2. **Combining Data** <br>
Combine all the data files within the above folders and format them with timestamps. This process will automatically eliminate any duplicate days in the data files.
    ```
    command/combine_data.sh
    ```

3. **Preprocessing Data** <br>
Reformat the combined data into a format that can be used for training the models.
    ```
    command/preprocess_data.sh
    ```

4. **Training the Models** <br>
Train the models using the prepared dataset and save the trained models locally.
    ```
    command/train.sh
    ```

***NOTICE:***
1. The data is input in the form of records at 5-minute intervals.
2. The data output involves predicting the next 2 hours based on the previous 12 hours.
3. If you need to adjust the number of days used for prediction or the number of days predicted, you can edit the following arguments in the `command/train.sh` file:
    ```
    --prev_day 12 \
    --pred_day 2 \
    ```


# 3. Step for using the model in production

Utilizing the models for prediction can be accomplished as outlined below:

1. **Preparing the Data** <br>
Organize the data into two distinct folders named "CPU" and "RAM."
    ```
    /data/test_data
    ```


2. **Generating Predictions** <br>
The model will automatically gather the required number of data entries in reverse order to fulfill the necessary prediction time frame, and then produce the results.
    ```
    command/combine_data.sh
    ```



***NOTICE:***
1. The data is input in the form of records at 5-minute intervals.
2. The data output involves predicting the next 2 hours based on the previous 12 hours.
3. If you need to adjust the number of days used for prediction or the number of days predicted, you can edit the following arguments in the `command/combine_data.sh` file:
    ```
    --prev_day 12 \
    --pred_day 2 \
    ```
