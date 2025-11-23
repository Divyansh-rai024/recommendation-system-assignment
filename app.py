# app.py
import streamlit as st
import pandas as pd
from query_functions import query_handling_using_LLM_updated
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import pickle
from pathlib import Path
import os

st.set_page_config(page_title="SHL Assessment Recommendation System", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>🧠 SHL Assessment Recommendation System</h1>
    <h4 style='text-align: center; color: #ccc;'>Find the best assessments based on your query using AI!</h4>
    <hr style="border: 1px solid #333;">
    """,
    unsafe_allow_html=True
)

# Load cached models & data locally for Streamlit (same startup logic as API)
CATALOG_FILE = "SHL_catalog.csv"
EMB_FILE = "corpus_embeddings.npy"
CORPUS_PICKLE = "corpus.pkl"

@st.cache_resource
def load_resources():
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    if not Path(CATALOG_FILE).exists():
        raise FileNotFoundError(f"{CATALOG_FILE} not found")
    catalog_df = pd.read_csv(CATALOG_FILE)
    # normalize
    catalog_df["Description"] = catalog_df["Description"].fillna("").astype(str)
    corpus = catalog_df.apply(query_functions.build_combined_text_from_row, axis=1).tolist() if hasattr(query_functions, 'build_combined_text_from_row') else catalog_df['Description'].tolist()
    # load embeddings if available, else compute
    if Path(EMB_FILE).exists() and Path(CORPUS_PICKLE).exists():
        emb = torch.from_numpy(np.load(EMB_FILE))
        with open(CORPUS_PICKLE, "rb") as f:
            corpus = pickle.load(f)
    else:
        emb = sbert.encode(corpus, convert_to_tensor=True)
        np.save(EMB_FILE, emb.cpu().numpy())
        with open(CORPUS_PICKLE, "wb") as f:
            pickle.dump(corpus, f)
    return sbert, None, catalog_df, corpus, emb

# lazy load
try:
    import query_functions as query_functions  # ensure imported for helper usage in load_resources
except Exception:
    import query_functions

sbert_model, gemini_model, catalog_df, corpus, corpus_embeddings = load_resources()

query = st.text_input("🔍 Enter your search query here:", placeholder="e.g. Python SQL coding test")

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("🤖 Thinking... Fetching the best matches for you!"):
            try:
                df = query_handling_using_LLM_updated(
                    query,
                    sbert_model=sbert_model,
                    gemini_model=gemini_model,
                    catalog_df=catalog_df,
                    corpus=corpus,
                    corpus_embeddings=corpus_embeddings,
                    top_k=10
                )

                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Ensure expected columns exist
                    display_cols = ["Assessment Name", "Skills", "Test Type", "Description", "Remote Testing Support", "Adaptive/IRT", "Duration", "URL"]
                    df = df[[c for c in display_cols if c in df.columns]]

                    # Make URLs clickable for streamlit
                    if "URL" in df.columns:
                        df["URL"] = df["URL"].apply(lambda x: f"[View]({x})" if pd.notna(x) and x else "")

                    st.success("✅ Here are your top assessment recommendations:")
                    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
                else:
                    st.warning("😕 No assessments matched your query. Try rephrasing it!")

            except Exception as e:
                st.error(f"🚨 Something went wrong: {e}")
