import pandas as pd
import matplotlib.pyplot as plt



def get_all_nan_values(companies_df):
    return companies_df.isna().sum()

def histogram_for_nan_values():
   all_nan_values = get_all_nan_values()
   plt.figure(figsize=(12, 6))
   plt.xlabel("Label")
   plt.ylabel("Number of nan values for each column")
   plt.hist(all_nan_values.index, all_nan_values)
   plt.title("Histogram for nan values for each column")
