from lib2to3.pytree import convert

import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-mpnet-base-v2')
taxonomy_df = pd.read_csv("./datasets/insurance_taxonomy - insurance_taxonomy.csv")

nlp = spacy.load("en_core_web_sm")
insurance_values = taxonomy_df["label"].tolist()

def clean_company_info(row):
   text = f"{row['niche']} {row['category']} {row['business_tags']} {row['description']}"
   doc = nlp(text)

   #I will not keep names of brands, locations, people
   clean_tokens = [token.text for token in doc if token.ent_type not in ["ORG", "PERSON", "GPE"]]
   return " ".join(clean_tokens)

def clean_info(row):
    text = f"{row['niche']} {row['category']} {row['business_tags']} {row['description']}".lower()
    noise = ['limited', 'llc', 'inc', 'corp', 'solutions', 'international', 'group', 'and', 'the']
    words = text.split()
    clean_words = [w for w in words if w not in noise and len(w) > 2]
    return " ".join(clean_words)

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


def classify_with_expansion(company_row, insurance_values, insurance_values_expanded_emb):
    clean_text_info = clean_company_info(company_row)
    company_emb = model.encode(clean_text_info, convert_to_tensor=True)

    semantic_search = util.semantic_search(company_emb, insurance_values_expanded_emb, top_k=2)

    if semantic_search[0]['score'] > 0.35:
        return insurance_values[semantic_search[0]['corpus_id']]
    else:
        return "Unclassified"


def expand_labels(label):
    context_map = {
        "Construction": "building builder site infrastructure foundation renovation",
        "Services": "maintenance repair professional operations",
        "Manufacturing": "factory plant production processing assembly industrial",
        "Agricultural": "farm crop harvest tractor machinery cultivation",
        "Medical": "health clinical doctor hospital treatment",
        "Digital": "software technology online internet cloud",
        "Food": "beverage kitchen bakery processing nutrition"
    }
    expanded = label
    for key, val in context_map.items():
        if key.lower() in label.lower():
            expanded += " " + val
    return expanded

expanded_insurance_values = [expand_labels(label) for label in insurance_values]
insurance_values_expanded_emb = model.encode(expanded_insurance_values, convert_to_tensor=True)

def classify_with_expanded_version(row):

    cleaned_row = clean_info(row)
    company_emb = model.encode(cleaned_row, convert_to_tensor=True)

    semantic_search =util.semantic_search(company_emb, insurance_values_expanded_emb, top_k=2)
    best_match = semantic_search[0][0]
    if  best_match ['score'] > 0.35:
        return insurance_values[best_match['corpus_id']]
    else:
        return "Unclassified"



import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('all-MiniLM-L6-v2')

def create_weighted_input(row):

    niche = (str(row['niche']) + " ") if pd.notna(row['niche']) else ""
    cat = (str(row['category']) + " ") if pd.notna(row['category']) else ""
    tags = (str(row['business_tags']) + " ") if pd.notna(row['business_tags']) else ""
    sect = (str(row['sector']) + " ") if pd.notna(row['sector']) else ""
    desc = str(row['description']) if pd.notna(row['description']) else ""

    weighted_text = (niche * 5) + (cat * 3) + (tags * 2) + (sect * 2) + desc
    return weighted_text



def model_with_weights(companies_df):
 companies_df['weighted_features'] = companies_df.apply(create_weighted_input, axis=1)

 company_embeddings = model.encode(
    companies_df['weighted_features'].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_tensor=True
)


 tax_embeddings = model.encode(expanded_insurance_values, convert_to_tensor=True)

 cos_scores = util.cos_sim(company_embeddings, tax_embeddings)

 best_scores, best_indices = torch.max(cos_scores, dim=1)
 companies_df['insurance_label'] = [
    insurance_values[idx]
    for idx, score in zip(best_indices, best_scores)
 ]

 companies_df.drop(columns=['weighted_features']).to_csv('final_weighted_classification3.csv', index=False)

def model_with_weights_score(companies_df):
     companies_df['weighted_features'] = companies_df.apply(create_weighted_input, axis=1)

     company_embeddings = model.encode(
         companies_df['weighted_features'].tolist(),
         batch_size=64,
         show_progress_bar=True,
         convert_to_tensor=True
     )

     tax_embeddings = model.encode(expanded_insurance_values, convert_to_tensor=True)

     cos_scores = util.cos_sim(company_embeddings, tax_embeddings)

     best_scores, best_indices = torch.max(cos_scores, dim=1)
     companies_df['insurance_label'] = [
         insurance_values[idx] if score > 0.35 else "Unclassified"
         for idx, score in zip(best_indices, best_scores)
     ]

     companies_df.drop(columns=['weighted_features']).to_csv('final_weighted_classification2.csv', index=False)