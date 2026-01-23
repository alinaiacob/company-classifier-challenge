import ast
import pandas as pd

companies_df = pd.read_csv("./datasets/ml_insurance_challenge.csv")

model_df = pd.read_csv(
    "results/more_labels/companies_predicted_labels_only_model-more-labels_model.csv"
)


model_df = model_df[[
    "description",
    "top_labels"
]].rename(columns={
    "top_labels": "insurance_label"
})

model_df = model_df.drop_duplicates(subset=["description"])


companies_df = companies_df.merge(
    model_df,
    on="description",
    how="left"
)



def extract_labels_from_list(text):
    if pd.isna(text):
        return None
    try:
        pairs = ast.literal_eval(text)
        labels = [p[0] for p in pairs]
        return " | ".join(labels)
    except:
        return None


companies_df["insurance_label"] = companies_df["insurance_label"].apply(extract_labels_from_list)

companies_df.to_csv(
    "./datasets/ml_insurance_challenge.csv",
    index=False
)

print(companies_df.head())
