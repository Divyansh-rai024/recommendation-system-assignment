from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle
import torch
import os
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
import query_functions as qf

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- FILE PATHS ----------
CATALOG_FILE = "SHL_catalog.csv"
EMB_FILE = "corpus_embeddings.npy"
CORPUS_PICKLE = "corpus.pkl"

# ---------- GLOBALS ----------
sbert_model = None
gemini_model = None
catalog_df = None
corpus = None
corpus_embeddings = None

_parse_duration_to_int = qf._parse_duration_to_int
_split_to_list = qf._split_to_list


# ============================================================
#                INITIALIZE EVERYTHING AT STARTUP
# ============================================================
@app.on_event("startup")
def load_everything():
    global sbert_model, catalog_df, corpus, corpus_embeddings, gemini_model

    logger.info("🔥 Loading Model & Embeddings...")

    # --------------- SBERT MODEL ----------------
    model_path = "./all-MiniLM-L6-v2"
    if not Path(model_path).exists():
        raise RuntimeError(f"SBERT directory not found: {model_path}")

    sbert_model = SentenceTransformer(model_path)
    logger.info("✅ SBERT loaded")

    # --------------- CATALOG ----------------
    if not Path(CATALOG_FILE).exists():
        raise RuntimeError("Catalog file SHL_catalog.csv missing")

    df = pd.read_csv(CATALOG_FILE)
    df["Duration"] = df["Duration"].apply(_parse_duration_to_int)
    df["Skills"] = df["Skills"].fillna("").astype(str)
    df["Description"] = df["Description"].fillna("").astype(str)
    df["Test Type"] = df["Test Type"].fillna("").astype(str)
    df["Remote Testing Support"] = df["Remote Testing Support"].fillna("").astype(str)
    df["Adaptive/IRT"] = df["Adaptive/IRT"].fillna("").astype(str)

    catalog_df = df
    logger.info("✅ Catalog loaded")

    # --------------- EMBEDDINGS ----------------
    if not Path(EMB_FILE).exists() or not Path(CORPUS_PICKLE).exists():
        raise RuntimeError(
            "❌ Missing embeddings! Run embed_precompute.py before deploying."
        )

    corpus_embeddings = torch.from_numpy(np.load(EMB_FILE))
    with open(CORPUS_PICKLE, "rb") as f:
        corpus = pickle.load(f)

    logger.info("✅ Corpus + Embeddings loaded")

    # --------------- OPTIONAL GEMINI ---------------
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel("gemini-1.5-pro")
            logger.info("✅ Gemini loaded")
        except:
            gemini_model = None
            logger.warning("⚠️ Gemini failed to initialize")


@app.get("/health")
def health():
    return {"status": "ok"}


class QueryRequest(BaseModel):
    query: str


@app.post("/recommend")
def recommend(req: QueryRequest):
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
        raise HTTPException(status_code=404, detail="No predictions available")

    return {"csv": pd.DataFrame(rows).to_csv(index=False)}
