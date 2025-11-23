# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import logging
import numpy as np
import pickle
import re
from pathlib import Path
import torch

import query_functions as qf

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sbert_model: Optional[SentenceTransformer] = None
gemini_model = None
catalog_df: Optional[pd.DataFrame] = None
corpus = None
corpus_embeddings = None

EMB_FILE = "corpus_embeddings.npy"
CORPUS_PICKLE = "corpus.pkl"
CATALOG_FILE = "SHL_catalog.csv"

# parse helpers (reuse from query_functions where possible)
_parse_duration_to_int = qf._parse_duration_to_int
_split_to_list = qf._split_to_list

@app.on_event("startup")
def startup_event():
    global sbert_model, gemini_model, catalog_df, corpus, corpus_embeddings

    logger.info("Startup: Loading models and catalog...")

    # SBERT
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Gemini config
    api_key = os.getenv("GEMINI_API_KEY", None)
    try:
        if api_key:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel("gemini-1.5-pro")
            logger.info("Gemini configured.")
        else:
            gemini_model = None
            logger.warning("GEMINI_API_KEY not set. Using semantic-only fallback.")
    except Exception as e:
        gemini_model = None
        logger.warning(f"Failed to configure Gemini: {e}. Falling back to semantic-only.")

    # Load catalog
    catalog_path = Path(CATALOG_FILE)
    if not catalog_path.exists():
        logger.error(f"Catalog file not found at {CATALOG_FILE}")
        raise FileNotFoundError(f"{CATALOG_FILE} required in working directory")

    catalog_df = pd.read_csv(CATALOG_FILE)
    # Normalize fields
    catalog_df["Duration"] = catalog_df["Duration"].apply(_parse_duration_to_int)
    catalog_df["Skills"] = catalog_df["Skills"].fillna("").astype(str)
    catalog_df["Test Type"] = catalog_df["Test Type"].fillna("").astype(str)
    catalog_df["Remote Testing Support"] = catalog_df["Remote Testing Support"].fillna("").astype(str)
    catalog_df["Adaptive/IRT"] = catalog_df["Adaptive/IRT"].fillna("").astype(str)
    catalog_df["Description"] = catalog_df["Description"].fillna("").astype(str)

    # Build corpus
    corpus = catalog_df.apply(qf.build_combined_text_from_row, axis=1).tolist()

    # Load or compute embeddings
    try:
        if Path(EMB_FILE).exists() and Path(CORPUS_PICKLE).exists():
            logger.info("Loading cached embeddings...")
            corpus_embeddings = torch.from_numpy(np.load(EMB_FILE))
            with open(CORPUS_PICKLE, "rb") as f:
                corpus = pickle.load(f)
            logger.info("Cached embeddings loaded.")
        else:
            logger.info("Encoding corpus embeddings...")
            corpus_embeddings = sbert_model.encode(corpus, convert_to_tensor=True)
            np.save(EMB_FILE, corpus_embeddings.cpu().numpy())
            with open(CORPUS_PICKLE, "wb") as f:
                pickle.dump(corpus, f)
            logger.info("Embeddings cached.")
    except Exception as e:
        logger.exception("Failed loading cached embeddings, computing anew...")
        corpus_embeddings = sbert_model.encode(corpus, convert_to_tensor=True)
        np.save(EMB_FILE, corpus_embeddings.cpu().numpy())
        with open(CORPUS_PICKLE, "wb") as f:
            pickle.dump(corpus, f)

    logger.info("Startup complete. Service ready.")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

class QueryRequest(BaseModel):
    query: str

@app.post("/recommend")
def recommend_assessments(request: QueryRequest):
    global sbert_model, gemini_model, catalog_df, corpus, corpus_embeddings
    if not sbert_model or catalog_df is None or corpus_embeddings is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        df = qf.query_handling_using_LLM_updated(
            request.query,
            sbert_model=sbert_model,
            gemini_model=gemini_model,
            catalog_df=catalog_df,
            corpus=corpus,
            corpus_embeddings=corpus_embeddings,
            top_k=10
        )

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No assessments found")

        # df columns are renamed inside query_functions to SHL-like display names
        # Convert DataFrame rows to required JSON structure
        results = []
        for _, row in df.iterrows():
            # row expected keys: 'Assessment Name', 'URL', 'Adaptive/IRT', 'Description', 'Duration', 'Remote Testing Support', 'Test Type', 'Skills'
            name = row.get("Assessment Name", "")
            url = row.get("URL", "")
            adaptive = qf.normalize_yes_no(row.get("Adaptive/IRT", ""))
            remote = qf.normalize_yes_no(row.get("Remote Testing Support", ""))
            duration = int(_parse_duration_to_int(row.get("Duration", 0)))
            test_type_list = qf.normalize_test_type_list(row.get("Test Type", ""))
            skills_list = _split_to_list(row.get("Skills", ""))

            results.append({
                "name": name,
                "url": url,
                "adaptive_support": adaptive,
                "description": str(row.get("Description", "")).strip(),
                "duration": duration,
                "remote_support": remote,
                "test_type": test_type_list,
                "skills": skills_list
            })

        return {"recommended_assessments": results}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Recommendation error")
        raise HTTPException(status_code=500, detail=str(e))

# Optional helper: produce CSV predictions for a list of queries (assignment requires CSV format)
class BatchQuery(BaseModel):
    queries: List[str]

@app.post("/export_predictions")
def export_predictions(batch: BatchQuery):
    """
    Given a list of queries, return CSV content with columns: Query,Assessment_url
    (For each query produce up to 10 rows with the top URLs)
    """
    rows = []
    for q in batch.queries:
        try:
            df = qf.query_handling_using_LLM_updated(
                q,
                sbert_model=sbert_model,
                gemini_model=gemini_model,
                catalog_df=catalog_df,
                corpus=corpus,
                corpus_embeddings=corpus_embeddings,
                top_k=10
            )
            if df is None or df.empty:
                continue
            for _, r in df.iterrows():
                url = r.get("URL", "")
                if url:
                    rows.append({"Query": q, "Assessment_url": url})
        except Exception as e:
            logger.warning(f"Skipping query {q} due to error: {e}")
    if not rows:
        raise HTTPException(status_code=404, detail="No predictions generated")
    out_df = pd.DataFrame(rows)
    csv_data = out_df.to_csv(index=False)
    return {"csv": csv_data}
