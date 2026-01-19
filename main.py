import pandas as pd
from sentence_transformers import SentenceTransformer, util
import spacy

from utils.text_processing import classify_company_sentence_transformer

nlp = spacy.load("en_core_web_")
companies_df = pd.read_csv("./datasets/ml_insurance_challenge.csv")
taxonomy_df = pd.read_csv("./datasets/insurance_taxonomy - insurance_taxonomy.csv")

#to gain more knowledge my data I will create a new column that will contain all the concatenation for all columns

# 'description', 'business_tags', 'sector', 'category', 'niche' - all cols
all_cols = companies_df.columns.tolist()
companies_df["all_info"] = (companies_df[all_cols] .fillna("").agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip())

insurance_values = taxonomy_df["label"].tolist()
companies_info_list = companies_df["all_info"].tolist()

print(companies_df["all_info"].head())

#sentence transformers
#load model
model = SentenceTransformer('all-mpnet-base-v2')

#labels embeddings
insurance_embeddings = model.encode(insurance_values, convert_to_tensor=True)

#in this way this model does not classify the company description well - most of the labels are not classified
#I consider using NER - because a lot of description of company contains representatives brands and this aspect should improve the accuracy for our classification
classify_company_sentence_transformer(companies_info_list, insurance_values, insurance_embeddings)


#Also I consider using a fallback method because for lots of companies - score is pretty low






