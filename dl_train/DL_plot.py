
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# 1. LOAD AND PLOT THE PREDICTIONS (TOP SUBPLOT)
###############################################################################

# Path to the final ensemble predictions CSV
filepath_out = r'C:\Users\Brandon\repositories\deep-early-warnings-pnas\test_models\may_fold_1500\data\ml_preds\ensemble_trend_probs_seizure_transition_1500_1.csv'

# Read in the CSV of predictions (header=None, 4 columns)
df_predictions = pd.read_csv(filepath_out, header=None)

###############################################################################
# 2. LOAD AND PLOT THE RAW EEG DATA (BOTTOM SUBPLOT)
###############################################################################

# Path to the EEG CSV you used to generate predictions
filepath_eeg = r'C:\Users\Brandon\repositories\deep-early-warnings-pnas\test_models\seizure\ensemble_trend_probs_seizure_transition_1500_1.csv'
df_eeg = pd.read_csv(filepath_eeg)

# We’ll assume your EEG CSV has columns: 'Time_s' and 'Voltage'.
time_s = df_eeg['Time_s']
voltage = df_eeg['Voltage']

###############################################################################
# 3. CREATE A FIGURE WITH TWO SUBPLOTS
###############################################################################

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=False)

# --------------------
# Top Subplot: Predictions
# --------------------
axes[0].plot(df_predictions.index, df_predictions[0], label='fold_prob')
axes[0].plot(df_predictions.index, df_predictions[1], label='hopf_prob')
axes[0].plot(df_predictions.index, df_predictions[2], label='branch_prob')
axes[0].plot(df_predictions.index, df_predictions[3], label='null_prob')
axes[0].set_title('Ensemble DL Predictions')
axes[0].set_xlabel('Prediction Index')
axes[0].set_ylabel('Probability')
axes[0].legend()
axes[0].grid(True)

# Mark the 1000th sample in the top figure (index 999) as transition
transition_prediction_index = 100  # 1000th sample in 1-based count
if transition_prediction_index < len(df_predictions):
    axes[0].axvline(transition_prediction_index, color='r', linestyle='--', 
                    label='Transition (1000th sample)')

    # Add or update legend if necessary
    axes[0].legend()



# --------------------
# Bottom Subplot: Raw EEG
# --------------------
axes[1].plot(time_s, voltage, color='purple', label='EEG Voltage')
axes[1].set_title('Raw EEG Data (1500 samples)')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Voltage')
axes[1].legend()
axes[1].grid(True)

# Mark the 1000th sample (index 999) with a vertical line
transition_sample_index = 999  # 1000th sample in 1-based counting
if transition_sample_index < len(time_s):
    transition_time = time_s[transition_sample_index]
    axes[1].axvline(transition_time, color='r', linestyle='--', label='Transition')

axes[1].legend()

plt.tight_layout()
plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# # Path to the final ensemble CSV you wrote out:
# filepath_out = r'C:\Users\Brandon\repositories\deep-early-warnings-pnas\test_models\may_fold_1500\data\ml_preds\ensemble_trend_probs_seizure_transition_1500_1.csv'

# # Read in the CSV (no headers, 4 columns of predictions)
# df_predictions = pd.read_csv(filepath_out, header=None)

# # Create a figure
# plt.figure(figsize=(8, 5))

# # Plot each of the four columns
# plt.plot(df_predictions.index, df_predictions[0], label='fold_prob')
# plt.plot(df_predictions.index, df_predictions[1], label='hopf_prob')
# plt.plot(df_predictions.index, df_predictions[2], label='branch_prob')
# plt.plot(df_predictions.index, df_predictions[3], label='null_prob')

# # Labeling
# plt.xlabel('Time Steps')
# plt.ylabel('Predictions')
# plt.title('Ensemble DL Predictions')
# plt.legend()
# plt.grid(True)
# plt.show()

