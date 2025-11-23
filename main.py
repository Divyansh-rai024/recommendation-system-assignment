# main.py (Lazy Load Version)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from pathlib import Path
import os
import torch
import logging
import query_functions as qf

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals (start empty)
sbert_model = None
gemini_model = None
catalog_df = None
corpus = None
corpus_embeddings = None

CATALOG_FILE = "SHL_catalog.csv"
EMB_FILE = "corpus_embeddings.npy"
CORPUS_PICKLE = "corpus.pkl"

_parse_duration_to_int = qf._parse_duration_to_int
_split_to_list = qf._split_to_list

# ---------- LAZY LOADER ----------
def load_everything():
    global sbert_model, gemini_model, catalog_df, corpus, corpus_embeddings

    if sbert_model is not None:
        return  # Already loaded

    logger.info("Lazy loading SBERT + catalog + embeddings...")

    # 1. Load Model
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 2. Load Catalog
    if not Path(CATALOG_FILE).exists():
        raise FileNotFoundError(f"{CATALOG_FILE} missing")

    catalog_df_local = pd.read_csv(CATALOG_FILE)
    catalog_df_local["Duration"] = catalog_df_local["Duration"].apply(_parse_duration_to_int)
    catalog_df_local["Skills"] = catalog_df_local["Skills"].fillna("").astype(str)
    catalog_df_local["Description"] = catalog_df_local["Description"].fillna("").astype(str)
    catalog_df_local["Test Type"] = catalog_df_local["Test Type"].fillna("").astype(str)
    catalog_df_local["Remote Testing Support"] = catalog_df_local["Remote Testing Support"].fillna("").astype(str)
    catalog_df_local["Adaptive/IRT"] = catalog_df_local["Adaptive/IRT"].fillna("").astype(str)

    corpus_local = catalog_df_local.apply(qf.build_combined_text_from_row, axis=1).tolist()

    # 3. Load or create embeddings
    if Path(EMB_FILE).exists() and Path(CORPUS_PICKLE).exists():
        logger.info("Loading cached embeddings...")
        corpus_embeddings_local = torch.from_numpy(np.load(EMB_FILE))
        with open(CORPUS_PICKLE, "rb") as f:
            corpus_local = pickle.load(f)
    else:
        logger.info("Encoding embeddings (first run)...")
        corpus_embeddings_local = sbert_model.encode(corpus_local, convert_to_tensor=True)
        np.save(EMB_FILE, corpus_embeddings_local.cpu().numpy())
        with open(CORPUS_PICKLE, "wb") as f:
            pickle.dump(corpus_local, f)

    # Assign globals
    catalog_df = catalog_df_local
    corpus = corpus_local
    corpus_embeddings = corpus_embeddings_local

    # Gemini (optional)
    api_key = os.getenv("GEMINI_API_KEY", None)
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel("gemini-1.5-pro")
        except:
            gemini_model = None

@app.get("/health")
def health():
    return {"status": "ok"}

class QueryRequest(BaseModel):
    query: str

@app.post("/recommend")
def recommend(req: QueryRequest):
    load_everything()  # <-- lazy load happens here

    df = qf.query_handling_using_LLM_updated(
        req.query,
        sbert_model=sbert_model,
        gemini_model=gemini_model,
        catalog_df=catalog_df,
        corpus=corpus,
        corpus_embeddings=corpus_embeddings,
        top_k=10
    )

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No matching assessments")

    results = []
    for _, row in df.iterrows():
        results.append({
            "name": row.get("Assessment Name", ""),
            "url": row.get("URL", ""),
            "adaptive_support": qf.normalize_yes_no(row.get("Adaptive/IRT", "")),
            "remote_support": qf.normalize_yes_no(row.get("Remote Testing Support", "")),
            "duration": int(_parse_duration_to_int(row.get("Duration", 0))),
            "description": str(row.get("Description", "")).strip(),
            "test_type": qf.normalize_test_type_list(row.get("Test Type", "")),
            "skills": _split_to_list(row.get("Skills", ""))
        })

    return {"recommended_assessments": results}

class BatchQuery(BaseModel):
    queries: List[str]

@app.post("/export_predictions")
def export_predictions(batch: BatchQuery):
    load_everything()

    rows = []
    for q in batch.queries:
        df = qf.query_handling_using_LLM_updated(
            q,
            sbert_model=sbert_model,
            gemini_model=gemini_model,
            catalog_df=catalog_df,
            corpus=corpus,
            corpus_embeddings=corpus_embeddings,
            top_k=10
        )
        if df is not None and not df.empty:
            for _, r in df.iterrows():
                url = r.get("URL", "")
                if url:
                    rows.append({"Query": q, "Assessment_url": url})

    if not rows:
        raise HTTPException(status_code=404, detail="No predictions")

    return {"csv": pd.DataFrame(rows).to_csv(index=False)}
