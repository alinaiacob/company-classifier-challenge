import pandas as pd
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import torch
from utils.text_processing import normalize_text

#aceasta solutie a fost gandita din considerentul in care si valorile disponibile din taxonomie  sa fie clasificate pe baza informatiilor deja existente in lista de companii (sectoare, business tags, niche)..
#solutia foloseste analiza de cluster ierarhica - in acest fel fiecare label din taxonomy va avea atribuit un sector si o categorie
#iar dupa comparatia  se va face intre informatiile companiei si ce label-uri au acelasi sector, categorie - in acest fel se produce o optimizare
#La final pentru fiecare companie obtinem un singur label recomandat, împreună cu scorul de similaritate (confidence), ceea ce permite atat interpretabilitate, cat si analiza a cazurilor de incertitudine.

companies_df = pd.read_csv("../datasets/ml_insurance_challenge.csv")
taxonomy_df = pd.read_csv("../datasets/insurance_taxonomy - insurance_taxonomy.csv")

companies_df["semantic_text"] = (
    companies_df["description"].fillna("") + " " +
    companies_df["business_tags"].fillna("") + " " +
    companies_df["category"].fillna("") + " " +
    companies_df["niche"].fillna("")
)

companies_df["semantic_text"] = companies_df["semantic_text"].apply(normalize_text)

taxonomy_df["semantic_text"] = taxonomy_df["label"]

model = SentenceTransformer("all-MiniLM-L6-v2")

company_embeddings = model.encode(
    companies_df["semantic_text"].tolist(),
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True
)


label_embeddings = model.encode(
    taxonomy_df["semantic_text"].tolist(),
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True
)


top_n = 5
taxonomy_sectors = []
taxonomy_categories = []

for i, label_emb in enumerate(label_embeddings):
    scores = util.cos_sim(label_emb, company_embeddings)[0]
    top_indices = scores.topk(top_n).indices.tolist()
    matched_companies = companies_df.iloc[top_indices]

    top_sector = Counter(matched_companies['sector']).most_common(1)[0][0]
    top_category = Counter(matched_companies['category']).most_common(1)[0][0]

    taxonomy_sectors.append(top_sector)
    taxonomy_categories.append(top_category)

taxonomy_df["sector_est"] = taxonomy_sectors
taxonomy_df["category_est"] = taxonomy_categories


sector_list = taxonomy_df["sector_est"].unique().tolist()
sector_embeddings = model.encode(sector_list, convert_to_tensor=True, normalize_embeddings=True)

category_list = taxonomy_df["category_est"].unique().tolist()
category_embeddings = model.encode(category_list, convert_to_tensor=True, normalize_embeddings=True)


sector_category_label_map = {}
for sector in sector_list:
    sector_categories = taxonomy_df[taxonomy_df["sector_est"] == sector]["category_est"].unique()
    sector_category_label_map[sector] = {}
    for category in sector_categories:
        labels_in_cat = taxonomy_df[
            (taxonomy_df["sector_est"] == sector) &
            (taxonomy_df["category_est"] == category)
            ].index.tolist()
        sector_category_label_map[sector][category] = labels_in_cat


results = []

top_n_labels = 1  # poți schimba la 1,2,3

results = []

for idx, row in companies_df.iterrows():
    company_emb = company_embeddings[idx]

    # Predict sector
    sector_scores = util.cos_sim(company_emb, sector_embeddings)[0]
    best_sector_idx = int(sector_scores.argmax())
    predicted_sector = sector_list[best_sector_idx]

    # Predict category within sector
    categories_in_sector = list(sector_category_label_map[predicted_sector].keys())
    cat_indices = [category_list.index(c) for c in categories_in_sector]
    cat_embs = category_embeddings[cat_indices]
    cat_scores = util.cos_sim(company_emb, cat_embs)[0]
    best_cat_idx = int(cat_scores.argmax())
    predicted_category = category_list[cat_indices[best_cat_idx]]

    # Predict top N labels within category
    label_indices = sector_category_label_map[predicted_sector][predicted_category]
    label_embs_candidates = label_embeddings[label_indices]
    label_scores = util.cos_sim(company_emb, label_embs_candidates)[0]

    top_indices = label_scores.topk(top_n_labels).indices.tolist()
    top_labels = [(taxonomy_df.iloc[label_indices[i]]["label"], float(label_scores[i])) for i in top_indices]

    results.append({
        "description": row.get("description", idx),
        "business_tags": row.get("business_tags"),
        "sector": row.get("sector"),
        "category": row.get("category"),
        "niche": row.get("niche"),
        "predicted_top_labels": top_labels
    })

final_df = pd.DataFrame(results)
final_df.to_csv("companies_predicted_hierarchical_more_labels.csv", index=False)
print(final_df.head(10))
