'''
Code to generate ensemble predictions from the DL classifiers
on a give time series of residuals

'''
import time
# Start timing
tic = time.time()


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gc
import numpy as np
import pandas as pd
import random
import sys
import itertools

import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime

random.seed(datetime.now())


'''
10 of the DL models are trained on training set data that are censored (padded) 
on the left hand side only, hence they are trained on data that always includes 
the transition.  The other 10 were trained on training set data that are 
censored on both left and right, hence they do not include the transition.
kk runs from 1 to 10 and denotes the index of the 10 models of each type.
model_type is 1 or 2.  1 denotes the model that is trained on data censored on 
both the left and right.  2 is the model trained on data censored on the left only. 

'''


# Filepath to residual time series to make predictions on 
filepath = r'F:\EEG_SeizureTransitions\eeg_signal.csv'

# Filepath to export ensemble DL predictions to
filepath_out = r'C:\Users\Brandon\repositories\deep-early-warnings-pnas\test_models\seizure\ensemble_trend_probs_seizure_transition_1500_1.csv'

# Parameters
window_size = 1500  # Sliding window size
step_size = 750     # Overlap between windows (e.g., 50% overlap)
downsample_factor = 10  # Downsample by selecting every nth sample

# Load and preprocess data
print("Loading data...")
df = pd.read_csv(filepath).dropna()
resids = df.iloc[:, 1].values.reshape(1,-1,1)
downsampled_data = resids[::downsample_factor]  # Downsample data
total_samples = len(downsampled_data)
seq_len = len(df)


print(f"Original size: {len(resids)}, Downsampled size: {total_samples}")

# Define function to generate sliding windows
def generate_windows(data, window_size, step_size):
    # Access the second dimension of the data (time series)
    data = data[0, :, 0]  # Reshape from (1, 3599999, 1) to (3599999,)

    for start in range(0, len(data) - window_size + 1, step_size):
        yield data[start:start + window_size].reshape(1, window_size, 1)

# Type of classifier to use (1500 or 500)
ts_len=1500

'''  
The following two parameters control how many sample points along the 
timeseries are used, and the length between them.  For instance, for an input 
time series equal to or less then length 1500, mult_factor=10 and 
pad_samples=150 means that it will do the unveiling in steps of 10 datapoints, 
at 150 evenly spaced points across the entire time series (150 x 10 = 1500).
Needs to be adjusted according to whether you are using the trained 500 length 
or 1500 length classifier.
'''

# Steps of datapoints in between each DL prediction
mult_factor = 10

# Total number of DL predictions to make
# Use 150 for length 1500 time series. Use 50 for length 500 time series.
pad_samples = 150



# # Load residual time series data
# df = pd.read_csv(filepath).dropna()
# resids = df['Voltage'].values.reshape(1,-1,1)
# # Length of inupt time series
# seq_len = len(df)



def get_dl_predictions(window, model_type, kk):
    
    '''
    Generate DL prediction time series on resids
    from DL classifier with type 'model_type' and index kk.
    '''
        
    # Setup file to store DL predictions
    predictions_file_name = r'C:\Users\Brandon\repositories\deep-early-warnings-pnas\dl_train\predictions\y_pred_{}_{}.csv'.format(kk,model_type)
    f1 = open(predictions_file_name,'w')

    # Load in specific DL classifier
    model_name = r'C:\Users\Brandon\repositories\deep-early-warnings-pnas\dl_train\best_models\best_model_{}_{}_len{}.pkl'.format(kk,model_type,ts_len)
    model = load_model(model_name)
    
    # Loop through each possible length of padding
    # Start with revelaing the DL algorith only the earliest points
    for pad_count in range(pad_samples-1, -1, -1):
    
        temp_ts = np.zeros((1,ts_len,1))
    
        ts_gap = ts_len-seq_len
        pad_length = mult_factor*pad_count
    
        if pad_length + ts_gap > ts_len:
            zero_range = ts_len
        else:
            zero_range = pad_length + ts_gap
        
        if zero_range == ts_len:
            # Set all ML predictions to zero
            y_pred = np.zeros(4).reshape(1,4)
        else:
            for j in range(0, zero_range):
                temp_ts[0,j] = 0
            for j in range(zero_range, ts_len):
                temp_ts[0,j] = resids[0,j-zero_range]
    
            # normalizing inputs: take averages, since the models were also trained on averaged data. 
            values_avg = 0.0
            count_avg = 0
            for j in range (0,ts_len):
                if temp_ts[0,j] != 0:
                    values_avg = values_avg + abs(temp_ts[0,j])
                    count_avg = count_avg + 1
            if count_avg != 0:
                values_avg = values_avg/count_avg
            for j in range (0,ts_len):
                if temp_ts[0,j] != 0:
                    temp_ts[0,j] = temp_ts[0,j]/values_avg
            
            # Compute DL prediction
            y_pred = model.predict(temp_ts)
            
                    
    
        # Write predictions to file
        #np.savetxt(f1, y_pred, delimiter=',')
        #print('Predictions computed for padding={}'.format(pad_count*mult_factor))
        
    # Delete model and do garbage collection to free up RAM
    tf.keras.backend.clear_session()
    if zero_range != ts_len:
        del model
    gc.collect()
    f1.close()
    
    return y_pred



# Compute DL predictions from all 20 trained models
# for model_type in [1,2]:                                
#     for kk in np.arange(1,11):
#         print('Compute DL predictions for model_type={}, kk={}'.format(
#             model_type,kk))
        
#         get_dl_predictions(resids, model_type, kk)

all_predictions = []

print("Starting sliding window predictions...")
for i, window in enumerate(generate_windows(downsampled_data, window_size, step_size)):
    print(f"Processing window {i + 1}/{(total_samples - window_size) // step_size + 1}")

    # Ensemble predictions
    ensemble_window_preds = []
    for model_type in [1, 2]:  # Model types
        for kk in range(1, 11):  # 10 models per type
            prediction = get_dl_predictions(window, model_type, kk)
            ensemble_window_preds.append(prediction)

    # Average predictions for the current window
    window_mean_prediction = np.mean(ensemble_window_preds, axis=0)
    all_predictions.append(window_mean_prediction)

# Combine all predictions into a single array
final_predictions = np.vstack(all_predictions)




# Compute average prediction among all 20 DL classifiers
# list_df_preds = []
# for model_type in [1,2]:
#     for kk in np.arange(1,11):
#         filename = r'C:\Users\Brandon\repositories\deep-early-warnings-pnas\dl_train\predictions\y_pred_{}_{}.csv'.format(kk,model_type)
#         df_preds = pd.read_csv(filename,header=None)
#         df_preds['time_index'] = df_preds.index
#         df_preds['model_type'] = model_type
#         df_preds['kk'] = kk
#         list_df_preds.append(df_preds)
    

# Save results to CSV
print("Saving predictions...")
np.savetxt(filepath_out, final_predictions, delimiter=',', fmt='%.4f')

toc = time.time()
print(f"Finished! Total elapsed time: {toc - tic:.2f} seconds")

# # Concatenate
# df_preds_all = pd.concat(list_df_preds).reset_index(drop=True)

# # Compute mean over all predictions
# df_preds_mean = df_preds_all.groupby('time_index').mean()

# # End timing
# toc = time.time()

# print(f"Elapsed time: {toc - tic:.4f} seconds")

# # Export predictions
# df_preds_mean[[0,1,2,3]].to_csv(filepath_out,index=False,header=False)




