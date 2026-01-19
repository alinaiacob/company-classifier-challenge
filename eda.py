import pandas as pd
import numpy as np
from utils.eda import get_all_nan_values, histogram_for_nan_values

companies_df = pd.read_csv("./datasets/ml_insurance_challenge.csv")

print(companies_df.describe())

columns = companies_df.columns.to_list()
print(columns)

all_nan = get_all_nan_values(companies_df)
print(type(all_nan))

histogram_for_nan_values(companies_df)



