import pandas as pd
import matplotlib.pyplot as plt



def get_all_nan_values(companies_df):
   # I keep only the columns with nan values to prevent the "noise"
    nan_values =  companies_df.isna().sum().sort_values(ascending=False)
    nan_values = nan_values[nan_values>0]
    return nan_values

def histogram_for_nan_values(companies_df):
   all_nan_values = get_all_nan_values(companies_df)
   plt.figure(figsize=(12, 6))
   plt.xlabel("Label")
   plt.ylabel("Number of nan values for each column")
   plt.bar(all_nan_values.index, all_nan_values.values)
   plt.xticks(rotation=45, ha="right")
   plt.title("Histogram for nan values for each column")

   plt.show()
