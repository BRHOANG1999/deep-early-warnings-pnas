#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 10:40:22 2020

Organise seizures data into transition sections (Hennekam et al. 2020)

@author: Thomas M. Bury
"""


import numpy as np
import pandas as pd

# Raw data (excel file)
xls = pd.ExcelFile('C:/Users/Brandon/repositories/deep-early-warnings-pnas/test_empirical/seizure/data/data_seizures.xlsx', engine='openpyxl')

# Import all columns from the specified sheet
df_seizure = pd.read_excel(xls, 'Sheet1')


df_transition_data = df_seizure.copy()  # Use the full datase
df_transition_data['tsid'] = 1  # Assign a single time series IDt


# Export transition data
df_transition_data.to_csv('C:/Users/Brandon/repositories/deep-early-warnings-pnas/test_empirical/seizure/data/data_transitions.csv',
                          index=False)



