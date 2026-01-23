import ast
import streamlit as st
import pandas as pd
import re
import numpy as np
import plotly.express as px

flat_df = pd.read_csv("results/more_labels/companies_predicted_labels_only_model-more-labels.csv")
hier_df = pd.read_csv("results/more_labels/companies_predicted_hierarchical_more_labels.csv")
mpnet_df = pd.read_csv("results/more_labels/companies_predicted_labels_only_model-more-labels_model.csv")


flat_df = flat_df.rename(columns={'top_labels': 'top_labels_flat'})
hier_df = hier_df.rename(columns={'top_labels': 'top_labels_hier'})
mpnet_df = mpnet_df.rename(columns={'top_labels': 'top_labels_mpnet'})

def extract_top1_score(text):
    if pd.isna(text) or text.strip() == "":
        return np.nan
    try:
        labels_list = ast.literal_eval(text)
        return float(labels_list[0][1])  # scorul primului label
    except:
        return np.nan

flat_df['top1_score'] = flat_df['top_labels_flat'].apply(extract_top1_score)
hier_df['top1_score'] = hier_df['top_labels_hier'].apply(extract_top1_score)
mpnet_df['top1_score'] = mpnet_df['top_labels_mpnet'].apply(extract_top1_score)



merged_df = flat_df.merge(
    hier_df[['description','top_labels_hier']],
    on='description'
).merge(
    mpnet_df[['description','top_labels_mpnet']],
    on='description'
)



st.title("Comparatie Top N Label Prediction per Company")
st.write("Compararea Flat, Hierarchical și MPNet pe baza Top N label-uri si scoruri cosine.")


page_size = 20
num_pages = (len(merged_df)//page_size) + 1
page = st.slider("Select page", 1, num_pages, 1)

start_idx = (page-1)*page_size
end_idx = start_idx+page_size
sample_df = merged_df.iloc[start_idx:end_idx]



st.subheader(f"Companii (page {page})")
st.dataframe(sample_df[[
    'description',
    'top_labels_flat',
    'top_labels_hier',
    'top_labels_mpnet'
]].rename(columns={
    'top_labels_flat': 'Top N Flat',
    'top_labels_hier': 'Top N Hierarchical',
    'top_labels_mpnet': 'Top N MPNet'
}))

st.title("Rezultatele obtinute pentru fiecare model")
st.write("Media scorului pentru primul label pentru modelul Flat", flat_df["top1_score"].mean())
st.write("Media scorului pentru primul label pentru modelul Hierarchical", hier_df["top1_score"].mean())
st.write("Media scorului pentru primul label pentru modelul Mpnet", mpnet_df["top1_score"].mean())


def extract_top1_tuple(text):
    if pd.isna(text) or text.strip() == "":
        return None
    try:
        labels_list = ast.literal_eval(text)
        if len(labels_list) == 0:
            return None

        return labels_list[0][0]
    except:
        return None

sample_df['top1_flat'] = sample_df['top_labels_flat'].apply(extract_top1_tuple)
sample_df['top1_hier'] = sample_df['top_labels_hier'].apply(extract_top1_tuple)
sample_df['top1_mpnet'] = sample_df['top_labels_mpnet'].apply(extract_top1_tuple)

st.subheader("Comparatie label principal (Top 1)")
st.write("Top 1 label identic Flat vs Hier:", (sample_df['top1_flat']==sample_df['top1_hier']).sum(), " / ", len(sample_df))
st.write("Top 1 label identic Flat vs MPNet:", (sample_df['top1_flat']==sample_df['top1_mpnet']).sum(), " / ", len(sample_df))
st.write("Top 1 label identic Hier vs MPNet:", (sample_df['top1_hier']==sample_df['top1_mpnet']).sum(), " / ", len(sample_df))



def extract_all_scores(text):
    if pd.isna(text) or text.strip() == "":
        return []
    try:
        labels_list = ast.literal_eval(text)
        return [s[1] for s in labels_list]  # toate scorurile
    except:
        return []

hist_df = pd.DataFrame({
    'Flat': sum(flat_df['top_labels_flat'].apply(extract_all_scores).tolist(), []),
    'MPNet': sum(mpnet_df['top_labels_mpnet'].apply(extract_all_scores).tolist(), [])
})

hist_df = hist_df.melt(var_name='Method', value_name='Score')
fig = px.histogram(hist_df, x='Score', color='Method', barmode='overlay', nbins=20,
                   title="Distribuția scorurilor cosine (Top N labels)")
fig.show()


def extract_scores(text):
    if pd.isna(text): return []
    return [float(s) for s in re.findall(r"\(([\d\.]+)\)", text)]


st.subheader("Exemple companii unde Top 1 label difera")
diff_df = sample_df[(sample_df['top1_flat']!=sample_df['top1_hier']) |
                    (sample_df['top1_flat']!=sample_df['top1_mpnet'])]

st.dataframe(diff_df[[
    'description',
    'top_labels_flat',
    'top_labels_hier',
    'top_labels_mpnet'
]].rename(columns={
    'top_labels_flat': 'Top N Flat',
    'top_labels_hier': 'Top N Hierarchical',
    'top_labels_mpnet': 'Top N MPNet'
}))
