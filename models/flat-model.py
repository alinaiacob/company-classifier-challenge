import pandas as pd
from sentence_transformers import SentenceTransformer, util
from utils.text_processing import normalize_text

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


#model = SentenceTransformer("all-MiniLM-L6-v2")

model = SentenceTransformer("all-mpnet-base-v2")

company_embeddings = model.encode(
    companies_df["semantic_text"].tolist(),
    convert_to_tensor=True,
    show_progress_bar=True
)

label_embeddings = model.encode(
    taxonomy_df["semantic_text"].tolist(),
    convert_to_tensor=True,
    show_progress_bar=True
)

results = []
top_n = 2

for i, company_emb in enumerate(company_embeddings):
    cosine_scores = util.cos_sim(company_emb, label_embeddings)[0]

    # extrage top N label-uri
    top_indices = cosine_scores.topk(top_n).indices.tolist()
    top_labels = [(taxonomy_df.iloc[j]["label"], float(cosine_scores[j])) for j in top_indices]


    # best_idx = int(cosine_scores.argmax())
    # best_label = taxonomy_df.iloc[best_idx]["label"]
    # confidence = float(cosine_scores[best_idx])

#description,business_tags,sector,category,niche
    results.append({
        "description": companies_df.iloc[i]["description"],
        "business_tags":companies_df.iloc[i]["business_tags"],
        "sector":companies_df.iloc[i]["sector"],
        "category":companies_df.iloc[i]["category"],
        "niche":companies_df.iloc[i]["niche"],
        "top_labels": top_labels
    })



final_df = pd.DataFrame(results)

final_df.to_csv(
    "companies_predicted_labels_only_model-more-labels_model.csv",
    index=False
)


print(final_df.head(10))
