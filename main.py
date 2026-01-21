import pandas as pd
from sentence_transformers import SentenceTransformer, util
import spacy
from utils.text_processing import classify_company_sentence_transformer
from utils.text_processing import expand_labels
from utils.text_processing import classify_with_expanded_version
from utils.text_processing import model_with_weights, model_with_weights_score
from utils.test_solution import calculate_heuristic_precision

#nlp = spacy.load("en_core_web")
companies_df = pd.read_csv("./datasets/ml_insurance_challenge.csv")
taxonomy_df = pd.read_csv("./datasets/insurance_taxonomy - insurance_taxonomy.csv")

#to gain more knowledge my data I will create a new column that will contain all the concatenation for all columns

# 'description', 'business_tags', 'sector', 'category', 'niche' - all cols
all_cols = companies_df.columns.tolist()
#companies_df["all_info"] = (companies_df[all_cols] .fillna("").agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip())

insurance_values = taxonomy_df["label"].tolist()
#companies_info_list = companies_df["all_info"].tolist()

#print(companies_df["all_info"].head())

#sentence transformers
#load model
#model = SentenceTransformer('all-mpnet-base-v2') this model is way to low and I choose to use a smaller model with

#labels embeddings
#insurance_embeddings = model.encode(insurance_values, convert_to_tensor=True)

#in this way this model does not classify the company description well - most of the labels are not classified
#I consider using NER - because a lot of description of company contains representatives brands and this aspect should improve the accuracy for our classification
#classify_company_sentence_transformer(companies_info_list, insurance_values, insurance_embeddings)



#expanded_insurance_values = [expand_labels(label) for label in insurance_values]
#print("expanded insurance values ", expanded_insurance_values)
#insurance_exapanded_embd = model.encode(expanded_insurance_values, convert_to_tensor=True)


#companies_df["insurance_label"] = companies_df.apply(classify_with_expanded_version, axis=1)
#companies_df.to_csv("./datasets/insurance_companies_with_labels.csv ", index=False)

#model_with_weights_score(companies_df)


final_weighted_classification = pd.read_csv("final_weighted_classification2.csv")

print(final_weighted_classification['insurance_label'].value_counts())

precision_score = calculate_heuristic_precision(final_weighted_classification)
print(f"Heuristic Precision: {precision_score:.2f}%") #with no - > score > 0.35 heuristic precision is 52.81% which is a pretty low score
#with weights score -> 43.85%









