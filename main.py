# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Optional
# import pandas as pd
# from query_functions import query_handling_using_LLM_updated
# from sentence_transformers import SentenceTransformer
# import os
# import torch
# import google.generativeai as genai
# from dotenv import load_dotenv
# import logging
# import numpy as np
# import pickle
# import re
# from pathlib import Path

# app = FastAPI()
# logging.basicConfig(level=logging.INFO)

# sbert_model: Optional[SentenceTransformer] = None
# gemini_model = None
# catalog_df: Optional[pd.DataFrame] = None
# corpus = None
# corpus_embeddings = None

# EMB_FILE = "corpus_embeddings.npy"
# CORPUS_PICKLE = "corpus.pkl"
# CATALOG_FILE = "SHL_catalog.csv"


# def _parse_duration_to_int(val) -> int:
#     if pd.isna(val):
#         return 0
#     if isinstance(val, (int, np.integer)):
#         return int(val)
#     s = str(val)
#     # find first integer in the string
#     m = re.search(r"(\d+)", s)
#     if m:
#         return int(m.group(1))
#     return 0


# def _split_to_list(field):
#     if pd.isna(field):
#         return []
#     if isinstance(field, list):
#         return field
#     s = str(field).strip()
#     if not s:
#         return []
#     # split on common separators
#     parts = re.split(r",|\||;|\/| and ", s)
#     return [p.strip() for p in parts if p.strip()]


# @app.on_event("startup")
# def startup_event():
#     global sbert_model, gemini_model, catalog_df, corpus, corpus_embeddings

#     load_dotenv()
#     api_key = os.getenv("GEMINI_API_KEY", None)

#     logging.info("🚀 Loading models and data...")

#     # Load Sentence Transformer once
#     sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

#     # Configure Gemini only if API key provided
#     if api_key:
#         genai.configure(api_key=api_key)
#         gemini_model = genai.GenerativeModel("gemini-1.5-pro")
#         logging.info("✅ Gemini configured.")
#     else:
#         gemini_model = None
#         logging.warning("⚠️ GEMINI_API_KEY not set. LLM features will be disabled or mocked.")

#     # Load and preprocess catalog data
#     catalog_path = Path(CATALOG_FILE)
#     if not catalog_path.exists():
#         logging.error(f"Catalog file not found at {CATALOG_FILE}")
#         raise FileNotFoundError(f"{CATALOG_FILE} required in working directory")

#     catalog_df = pd.read_csv(CATALOG_FILE)

#     # Ensure Duration numeric and normalize Skills/Test Type fields
#     catalog_df["Duration"] = catalog_df["Duration"].apply(_parse_duration_to_int)
#     catalog_df["Skills"] = catalog_df["Skills"].fillna("").astype(str)
#     catalog_df["Test Type"] = catalog_df["Test Type"].fillna("").astype(str)
#     catalog_df["Remote Testing Support"] = catalog_df["Remote Testing Support"].fillna("").astype(str)
#     catalog_df["Adaptive/IRT"] = catalog_df["Adaptive/IRT"].fillna("").astype(str)
#     catalog_df["Description"] = catalog_df["Description"].fillna("").astype(str)

#     def combine_row(row):
#         parts = [
#             str(row["Assessment Name"]),
#             str(row["Duration"]),
#             str(row["Remote Testing Support"]),
#             str(row["Adaptive/IRT"]),
#             str(row["Test Type"]),
#             str(row["Skills"]),
#             str(row["Description"]),
#         ]
#         return ' '.join([p for p in parts if p and p != "nan"])

#     catalog_df['combined'] = catalog_df.apply(combine_row, axis=1)
#     corpus = catalog_df['combined'].tolist()

#     # Try to load cached embeddings; otherwise compute and cache them
#     try:
#         if Path(EMB_FILE).exists() and Path(CORPUS_PICKLE).exists():
#             logging.info("Loading cached embeddings...")
#             corpus_embeddings = torch.from_numpy(np.load(EMB_FILE))
#             with open(CORPUS_PICKLE, "rb") as f:
#                 corpus = pickle.load(f)
#             logging.info("Loaded cached embeddings.")
#         else:
#             logging.info("Encoding corpus and caching embeddings...")
#             corpus_embeddings = sbert_model.encode(corpus, convert_to_tensor=True)
#             # save numpy array version
#             np.save(EMB_FILE, corpus_embeddings.cpu().numpy())
#             with open(CORPUS_PICKLE, "wb") as f:
#                 pickle.dump(corpus, f)
#             logging.info("Caching complete.")
#     except Exception as e:
#         logging.warning(f"Embedding cache load failed: {e}. Recomputing embeddings...")
#         corpus_embeddings = sbert_model.encode(corpus, convert_to_tensor=True)
#         np.save(EMB_FILE, corpus_embeddings.cpu().numpy())
#         with open(CORPUS_PICKLE, "wb") as f:
#             pickle.dump(corpus, f)

#     logging.info("✅ Startup complete.")


# @app.get("/health")
# def health_check():
#     return {"status": "healthy"}


# # Request body
# class QueryRequest(BaseModel):
#     query: str


# # Response model
# class Assessment(BaseModel):
#     assessment_name: str
#     url: str
#     adaptive_support: str
#     description: str
#     duration: int
#     remote_support: str
#     test_type: List[str]
#     skills: List[str]


# class RecommendationResponse(BaseModel):
#     recommended_assessments: List[Assessment]


# @app.post("/recommend", response_model=RecommendationResponse)
# def recommend_assessments(request: QueryRequest):
#     global sbert_model, gemini_model, catalog_df, corpus, corpus_embeddings

#     if sbert_model is None or catalog_df is None or corpus_embeddings is None:
#         raise HTTPException(status_code=500, detail="Service not ready. Models/data not loaded.")

#     try:
#         # Call query handler in query_functions. It should return a pandas DataFrame.
#         df: pd.DataFrame = query_handling_using_LLM_updated(
#             request.query,
#             model=sbert_model,
#             gemini_model=gemini_model,
#             catalog_df=catalog_df,
#             corpus=corpus,
#             corpus_embeddings=corpus_embeddings
#         )

#         if df is None or (isinstance(df, pd.DataFrame) and df.empty):
#             raise HTTPException(status_code=404, detail="No assessments found.")

#         results = []
#         for _, row in df.iterrows():
#             duration_val = _parse_duration_to_int(row.get("Duration", 0))
#             test_type_list = _split_to_list(row.get("Test Type", ""))
#             skills_list = _split_to_list(row.get("Skills", ""))

#             results.append({
#                 "assessment_name": row.get("Assessment Name", ""),
#                 "url": row.get("URL", ""),
#                 "adaptive_support": row.get("Adaptive/IRT", ""),
#                 "description": row.get("Description", ""),
#                 "duration": int(duration_val),
#                 "remote_support": row.get("Remote Testing Support", ""),
#                 "test_type": test_type_list,
#                 "skills": skills_list
#             })

#         return {"recommended_assessments": results}

#     except HTTPException:
#         raise
#     except Exception as e:
#         logging.exception("Error while generating recommendations")
#         raise HTTPException(status_code=500, detail=str(e))

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
