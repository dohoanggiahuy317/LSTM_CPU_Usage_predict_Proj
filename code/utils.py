import os
import numpy as np
import tensorflow as tf




#---------------------------------------------------------------------------------------------------------------------
# CREATE TRAIN-TEST SPLIT (80:20)
#---------------------------------------------------------------------------------------------------------------------
def np_array_convert(dataset, prev, pred):
    dataX, dataY = [], []

    for col_index in range( len(dataset.columns) ):
        col = dataset[dataset.columns[col_index]]
        
        for i in range(len(dataset) - prev - pred + 1):
            a = col.iloc[i:(i+prev)]
            dataX.append(a)
            dataY.append(col.iloc[i + prev: i + prev + pred])

    return (np.array(dataX), np.array(dataY))




#-----------------------------------------------------------
# USING A LSTM MODEL FOR PREDICTION ON TIME SERIES
#-----------------------------------------------------------
def model_setup(x_train_scale, pred_day):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(512,input_shape=(x_train_scale.shape[1], 1),return_sequences=True))
    model.add(tf.keras.layers.LSTM(512,return_sequences=False))
    model.add(tf.keras.layers.Dense(pred_day))

    # model.summary()

    model.compile(loss='mean_absolute_error', optimizer= tf.keras.optimizers.Adam())

    return model


#-----------------------------------------------------------
# DEFINING CALLBACKS
#-----------------------------------------------------------
def callback_setup():
    es = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=8, verbose=1, restore_best_weights=True)
    lr_red = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, verbose=1, min_lr=0.0000001,)


    checkpoint_path = "./checkpoint/cp"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq=999999)
    
    return [es , lr_red, cp_callback]