import pandas as pd

# Load forced and null predictions
df_ml_forced = pd.read_csv('C:/Users/Brandon/repositories/deep-early-warnings-pnas/test_empirical/anoxia/data/ml_preds/parsed/df_ml_forced.csv')
df_ml_null = pd.read_csv('C:/Users/Brandon/repositories/deep-early-warnings-pnas/test_empirical/anoxia/data/ml_preds/parsed/df_ml_null.csv')

import matplotlib.pyplot as plt

# Filter data for a specific tsid and variable
tsid = 1
variable = 'Mo'
df_filtered = df_ml_forced[(df_ml_forced['tsid'] == tsid) & (df_ml_forced['Variable label'] == variable)]

# Plot probabilities
plt.figure(figsize=(10, 6))
plt.plot(df_filtered['fold_prob'], label='Fold', marker='o')
plt.plot(df_filtered['hopf_prob'], label='Hopf', marker='x')
plt.plot(df_filtered['branch_prob'], label='Branch', marker='s')
plt.plot(df_filtered['null_prob'], label='Null', marker='d')
plt.plot(df_filtered['bif_prob'], label='Bifurcation', linestyle='--')

# Add labels and legend
plt.title(f'ML Prediction Probabilities Over Time for tsid {tsid} ({variable})')
plt.xlabel('Time (Age [ka BP])')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()

