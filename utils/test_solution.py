import pandas as pd
import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

noise_words = {'services', 'service', 'systems', 'system', 'products', 'product', 'manufacturing', 'industry', 'company', 'and', 'with', 'for'}

def get_essential_keywords(text):
    if pd.isna(text): return set()
    # Curățăm textul și păstrăm doar rădăcinile cuvintelor importante
    text = re.sub(r'[^a-z\s]', '', str(text).lower())
    words = text.split()
    return {stemmer.stem(w) for w in words if w not in noise_words and len(w) > 2}


def calculate_heuristic_precision(df):

    def check_anchor(row):
        company_essence = get_essential_keywords(f"{row['niche']} {row['category']} {row['business_tags']}")

        assigned_labels = [l.strip() for l in row['insurance_label'].split(',')]
        for label in assigned_labels:
            label_keywords = get_essential_keywords(label)
            if any(kw in company_essence for kw in label_keywords):
                return True
        return False

    if len(df) == 0: return 0

    df['is_valid'] = df.apply(check_anchor, axis=1)
    precision = df['is_valid'].mean()
    return precision * 100

def calculate_heuristic_precision_with_score (df):
    classified_df = df[df['insurance_label'] != "Unclassified"].copy()
    def check_anchor(row):
        company_essence = get_essential_keywords(f"{row['niche']} {row['category']} {row['business_tags']}")

        assigned_labels = [l.strip() for l in row['insurance_label'].split(',')]
        for label in assigned_labels:
            label_keywords = get_essential_keywords(label)
            if any(kw in company_essence for kw in label_keywords):
                return True
        return False

    if len(df) == 0: return 0

    classified_df['is_valid'] = classified_df.apply(check_anchor, axis=1)
    precision = df['is_valid'].mean()
    return precision * 100


def debug_misclassifications(df, taxonomy_expanded_dict):
    """
    Analizează discrepanța dintre Modelul Semantic și Euristica de Cuvinte Cheie.
    """
    results = []

    # Filtrăm doar firmele clasificate
    classified = df[df['insurance_label'] != "Unclassified"].copy()

    for _, row in classified.iterrows():
        # 1. Extragem cuvinte cheie din firmă (Nișă + Categorie + Tags)
        company_text = f"{row['niche']} {row['category']} {row['business_tags']}"
        company_keywords = get_stemmed_keywords(company_text)

        # 2. Extragem cuvinte cheie din eticheta atribuită (folosind și varianta extinsă)
        label_assigned = row['insurance_label']
        # Luăm și sinonimele pe care le-am definit în taxonomia extinsă
        expanded_info = taxonomy_expanded_dict.get(label_assigned, label_assigned)
        label_keywords = get_essential_keywords(expanded_info)

        # 3. Verificăm suprapunerea
        intersection = company_keywords.intersection(label_keywords)
        has_anchor = len(intersection) > 0

        if not has_anchor:
            results.append({
                'niche': row['niche'],
                'assigned_label': label_assigned,
                'company_keywords': list(company_keywords),
                'label_keywords': list(label_keywords),
                'potential_reason': "Lexical Gap (Sinonime)" if row.get('score', 0) > 0.4 else "Semantic Drift (Zgomot)"
            })

    return pd.DataFrame(results)