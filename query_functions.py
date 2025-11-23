
# query_functions.py
import json
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import requests
from sentence_transformers import util
import torch
import os
from typing import List, Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def extract_url_from_text(text: str) -> Optional[str]:
    match = re.search(r'(https?://[^\s,]+)', text)
    return match.group(1) if match else None

def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, headers={'User-Agent': "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join(soup.get_text().split())
    except Exception as e:
        logger.warning(f"Failed to extract text from URL {url}: {e}")
        return ""

def _parse_duration_to_int(val) -> int:
    if pd.isna(val):
        return 0
    if isinstance(val, (int, np.integer)):
        return int(val)
    s = str(val)
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 0

def _split_to_list(field: Any) -> List[str]:
    if pd.isna(field):
        return []
    if isinstance(field, list):
        return [str(x).strip() for x in field if str(x).strip()]
    s = str(field).strip()
    if not s:
        return []
    parts = re.split(r",|\||;|\/| and ", s)
    return [p.strip() for p in parts if p.strip()]

def normalize_yes_no(val: Any) -> str:
    if pd.isna(val):
        return "No"
    s = str(val).strip().lower()
    if any(x in s for x in ["yes", "y", "true", "supported", "available", "remote"]):
        return "Yes"
    if any(x in s for x in ["no", "n", "false", "not"]):
        return "No"
    # fallback: if string long treat as Yes? default No
    return "Yes" if len(s) > 0 else "No"

def normalize_test_type_list(field: Any) -> List[str]:
    parts = _split_to_list(field)
    out = []
    for p in parts:
        pl = p.strip().lower()
        if "knowledge" in pl or "k-" in pl or pl == "k":
            out.append("K")
        elif "personality" in pl or "p-" in pl or pl == "p":
            out.append("P")
        elif "ability" in pl:
            out.append("A")
        else:
            # if single-letter token like 'K,P' handle it
            if pl in ["k", "p", "a"]:
                out.append(pl.upper())
            else:
                # keep original token trimmed (but uppercase first letter)
                out.append(p.strip())
    # remove duplicates preserving order
    seen = set()
    res = []
    for v in out:
        if v not in seen:
            seen.add(v)
            res.append(v)
    return res

def sbert_search(query: str,
                 sbert_model,
                 corpus_embeddings: torch.Tensor,
                 corpus: List[str],
                 top_k: int = 10):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_k = min(top_k, len(corpus))
    top_results = torch.topk(cosine_scores, k=top_k)
    indices = top_results[1].cpu().numpy().tolist()
    scores = top_results[0].cpu().numpy().tolist()
    return indices, scores

def safe_extract_json_array(text: str) -> Optional[List[Dict]]:
    """
    Try to extract the first JSON array found in text. Returns None if not found or invalid.
    """
    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return None
        arr_text = match.group()
        return json.loads(arr_text)
    except Exception as e:
        logger.warning(f"Failed to parse JSON array from LLM output: {e}")
        return None

def build_combined_text_from_row(row: pd.Series) -> str:
    parts = [
        str(row.get("Assessment Name", "")),
        str(row.get("Duration", "")),
        str(row.get("Remote Testing Support", "")),
        str(row.get("Adaptive/IRT", "")),
        str(row.get("Test Type", "")),
        str(row.get("Skills", "")),
        str(row.get("Description", "")),
    ]
    return ' '.join([p for p in parts if p and p != "nan"])

def filter_with_llm(gemini_model, user_query: str, top_results: List[Dict]) -> List[Dict]:
    """
    Calls gemini_model to filter/refine top_results. Expects gemini_model.generate_content(prompt).
    Returns filtered list (1..10). If LLM fails, returns original top_results.
    """
    try:
        # Build a plain top_results string to pass to LLM
        top_json = json.dumps(top_results, indent=2, default=lambda x: x if not isinstance(x, (np.ndarray,)) else x.tolist())
        prompt = f"""
You are given a user query and a list of candidate SHL assessments (as JSON).
Your job: from the candidate list, select only the assessments that match the user's intent,
based on duration, required skills, remote/adaptive requirements, and test type.
Return a JSON array (list) of candidate objects from the input (no extra fields).
Return minimum 1 and maximum 10 items. Preserve original fields.

User query:
\"\"\"{user_query}\"\"\"

Candidates:
{top_json}

Return only the JSON array.
"""
        resp = gemini_model.generate_content(prompt)
        if not resp or not getattr(resp, "text", None):
            logger.warning("Gemini returned no text. Falling back to semantic ranking.")
            return top_results[:max(1, min(10, len(top_results)))]
        out_text = resp.text.strip()
        parsed = safe_extract_json_array(out_text)
        if parsed is None or len(parsed) == 0:
            logger.warning("Gemini output did not contain valid JSON array. Falling back to semantic ranking.")
            return top_results[:max(1, min(10, len(top_results)))]
        # Ensure we return the candidates as dictionaries with expected fields
        return parsed[:10]
    except Exception as e:
        logger.exception("LLM filtering failed, falling back to semantic rank")
        return top_results[:max(1, min(10, len(top_results)))]

def find_assessments(user_query: str,
                     sbert_model,
                     gemini_model,
                     catalog_df: pd.DataFrame,
                     corpus: List[str],
                     corpus_embeddings: torch.Tensor,
                     k: int = 10) -> List[Dict]:
    # semantic search
    indices, scores = sbert_search(user_query, sbert_model, corpus_embeddings, corpus, top_k=k)
    results = []
    for idx, score in zip(indices, scores):
        r = catalog_df.iloc[int(idx)]
        results.append({
            "Assessment Name": r.get("Assessment Name", ""),
            "Skills": r.get("Skills", ""),
            "Test Type": r.get("Test Type", ""),
            "Description": r.get("Description", ""),
            "Remote Testing Support": r.get("Remote Testing Support", ""),
            "Adaptive/IRT": r.get("Adaptive/IRT", ""),
            "Duration": r.get("Duration", ""),
            "URL": r.get("URL", ""),
            "Score": float(score)
        })
    # If gemini_model provided, use it to refine results; otherwise return semantic
    if gemini_model is not None:
        filtered = filter_with_llm(gemini_model, user_query, results)
        # filter_with_llm returns dicts maybe missing Score; if missing, keep Score from original matches where possible
        # ensure we return list of dicts
        return filtered
    else:
        return results

def format_for_output(results: List[Dict]) -> List[Dict]:
    out = []
    for row in results:
        duration = _parse_duration_to_int(row.get("Duration", 0))
        test_type_list = normalize_test_type_list(row.get("Test Type", ""))
        skills_list = _split_to_list(row.get("Skills", ""))
        adaptive = normalize_yes_no(row.get("Adaptive/IRT", ""))
        remote = normalize_yes_no(row.get("Remote Testing Support", ""))
        out.append({
            "name": row.get("Assessment Name", ""),
            "url": row.get("URL", ""),
            "adaptive_support": adaptive,
            "description": str(row.get("Description", "")).strip(),
            "duration": int(duration),
            "remote_support": remote,
            "test_type": test_type_list,
            "skills": skills_list,
            # keep score optional for debugging
            **({"score": float(row.get("Score"))} if "Score" in row else {})
        })
    return out

def query_handling_using_LLM_updated(user_input: str,
                                    sbert_model,
                                    gemini_model,
                                    catalog_df: pd.DataFrame,
                                    corpus: List[str],
                                    corpus_embeddings: torch.Tensor,
                                    top_k: int = 10):
    """
    Main entrypoint. Returns pandas.DataFrame or empty DataFrame on failure.
    """
    logger.info("Running query handler")
    if not isinstance(catalog_df, pd.DataFrame):
        raise ValueError("catalog_df must be a pandas DataFrame")

    # If input contains a URL, extract page text and append
    url = extract_url_from_text(user_input)
    if url:
        page_text = extract_text_from_url(url)
        if page_text:
            user_input = user_input + " " + page_text

    # Use SBERT to find candidates
    candidates = find_assessments(user_input, sbert_model, gemini_model, catalog_df, corpus, corpus_embeddings, k=top_k)
    if not candidates:
        return pd.DataFrame()

    # If gemini did the filtering, candidates may already be refined
    formatted = format_for_output(candidates)

    # Convert to DataFrame
    df = pd.DataFrame(formatted)
    # For consistency with existing UI, rename fields to readable column names optionally
    if not df.empty:
        # Create display-friendly columns if present
        rename_map = {
            "name": "Assessment Name",
            "skills": "Skills",
            "test_type": "Test Type",
            "description": "Description",
            "remote_support": "Remote Testing Support",
            "adaptive_support": "Adaptive/IRT",
            "duration": "Duration",
            "url": "URL",
            "score": "Score"
        }
        df.rename(columns=rename_map, inplace=True)
    return df
