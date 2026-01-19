import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-mpnet-base-v2')

nlp = spacy.load("en_core_web_sm")

def clean_company_info(row):
   text = f"{row['niche']} {row['category']} {row['business_tags']} {row['description']}"
   doc = nlp(text)

   #I will not keep names of brands, locations, people
   clean_tokens = [token.text for token in doc if token.ent_type not in ["ORG", "PERSON", "GPE"]]
   return " ".join(clean_tokens)


def classify_company_sentence_transformer(companies_info_list, insurance_values, insurance_embeddings):
    for company_info in companies_info_list:
        company_info_embeddings = model.encode(company_info, convert_to_tensor=True)

        # semantic search
        best_matches = util.semantic_search(company_info_embeddings, insurance_embeddings, top_k=3)[0]

        if best_matches[0]["score"] > 0.45:
            best_label = insurance_values[best_matches[0]["corpus_id"]]
        else:
            best_label = "Unclassified"
        print("Company info -", company_info)
        print("Best label - ", best_label)
        print("------------------------------------------------")


def classify_with_expansion(company_row, insurance_values, insurance_embeddings):
    clean_text_info = clean_company_info(company_row)