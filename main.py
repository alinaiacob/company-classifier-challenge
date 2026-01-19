import pandas as pd
from sentence_transformers import SentenceTransformer, util
import spacy

nlp = spacy.load("en_core_web_")
companies_df = pd.read_csv("./datasets/ml_insurance_challenge.csv")
taxonomy_df = pd.read_csv("./datasets/insurance_taxonomy - insurance_taxonomy.csv")

#to gain more knowledge my data I will create a new column that will contain all the preexistent column

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

for company_info in companies_info_list:
    company_info_embeddings = model.encode(company_info, convert_to_tensor=True)

    #semantic search
    best_matches = util.semantic_search(company_info_embeddings, insurance_embeddings, top_k = 3)[0]

    if best_matches[0]["score"] > 0.45:
        best_label = insurance_values[best_matches[0]["corpus_id"]]
    else:
        best_label = "Unclassified"
    print("Company info -", company_info)
    print("Best label - ", best_label)
    print("------------------------------------------------")

    #in this way this model does not classify the company description well - most of the labels is not classified
    #I consider using NER - because a lot of description of company contains representatives brands and this aspect should improve the accuracy for our classification



