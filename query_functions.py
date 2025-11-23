# import json
# import numpy as np
# import pandas as pd
# import re
# from bs4 import BeautifulSoup
# import requests
# from sentence_transformers import SentenceTransformer,util
# import torch
# import google.generativeai as genai
# from dotenv import load_dotenv
# import os

# catalog_df = pd.read_csv("SHL_catalog.csv")

# def combine_row(row):
#     parts = [
#         str(row["Assessment Name"]),
#         str(row["Duration"]),
#         str(row["Remote Testing Support"]),
#         str(row["Adaptive/IRT"]),
#         str(row["Test Type"]),
#         str(row["Skills"]),
#         str(row["Description"]),
#     ]
#     return ' '.join(parts)

# catalog_df['combined'] = catalog_df.apply(combine_row,axis=1)

# corpus = catalog_df['combined'].tolist()

# model = SentenceTransformer('all-MiniLM-L6-v2')

# corpus_embeddings = model.encode(corpus,convert_to_tensor=True)

# def extract_url_from_text(text):
#     match = re.search(r'(https?://[^\s,]+)', text)
#     if match:
#         return match.group(1)
#     return None

# def extract_text_from_url(url):
#     try:
#         response = requests.get(url,headers={'User-Agent':"Mozilla/5.0"})
#         soup = BeautifulSoup(response.text,'html.parser')
#         return ' '.join(soup.get_text().split())
#     except Exception as e:
#         return f"Error:{e}"

# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")


# genai.configure(api_key=api_key)

# gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# def extract_features_with_llm(user_query):
#     prompt = f"""
# You are an intelligent assistant helping to recommend SHL assessments.

# The input below may be:
# 1. A natural language query describing assessment needs (e.g., "Need a Python test under 60 minutes").
# 2. A job description (JD) pasted directly.
# 3. A job description URL (already converted into text outside this function).
# 4. A combination of user query + JD.

# Your task is to extract and summarize key hiring features from the input. Look for and include the following **if available**:

# - Job Title  
# - Duration of Test  
# - Remote Testing Support (Yes/No)  
# - Adaptive/IRT Format (Yes/No)  
# - Test Type  
# - Skills Required  
# - Any other relevant hiring context

# Format your response as a **single line** like this:

# `<Job Title> <Duration> <Remote Support> <Adaptive> <Test Type> <Skills> <Other Info>`

# Skip any fields not mentioned — do not include placeholders or "N/A".

# ---
# Input:
# {user_query}

# Only return the final, clean sentence — no explanations.
# """

#     response = gemini_model.generate_content(prompt)
#     return response.text.strip()

# def find_assessments(user_query,k=5):
#     query_embedding = model.encode(user_query, convert_to_tensor = True)
#     cosine_scores = util.cos_sim(query_embedding,corpus_embeddings)[0]
#     top_k = min(k,len(corpus))
#     top_results = torch.topk(cosine_scores,k=top_k)
#     results = []
#     for score, idx in zip(top_results[0], top_results[1]):
#         idx = idx.item()
#         result = {
#             "Assessment Name": catalog_df.iloc[idx]['Assessment Name'],
#             "Skills": catalog_df.iloc[idx]['Skills'],
#             "Test Type": catalog_df.iloc[idx]['Test Type'],
#             "Description": catalog_df.iloc[idx]['Description'],
#             "Remote Testing Support": catalog_df.iloc[idx]['Remote Testing Support'],
#             "Adaptive/IRT": catalog_df.iloc[idx]['Adaptive/IRT'],
#             "Duration": catalog_df.iloc[idx]['Duration'],
#             "URL": catalog_df.iloc[idx]['URL'],
#             "Score": round(score.item(), 4)
#         }
#         results.append(result)
#     return results

# def convert_numpy(obj):
#     if isinstance(obj, (np.integer, np.int64)):
#         return int(obj)
#     elif isinstance(obj, (np.floating, np.float64)):
#         return float(obj)
#     elif isinstance(obj, (np.ndarray,)):
#         return obj.tolist()
#     else:
#         raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# def filter_relevant_assessments_with_llm(user_query, top_results):
#     prompt = f"""
# You are helping to refine assessment recommendations based on user needs.

# A user has entered the following query:
# "{user_query}"

# You are given 10 or less assessments retrieved using semantic similarity. 
# Your task is to go through each assessment and determine if it truly matches the user’s intent, based on the following:
# - Duration match (e.g., if the user wants "< 40 mins", exclude longer ones)
# - Skills match (e.g., user wants "Python" but test is on "Excel", reject it)
# - Remote support, Adaptive format, Test type, or any clearly stated requirement
# - Ignore irrelevant matches, even if score is high

# Return only the assessments that are **highly relevant** to the query. 
# Use your understanding of language and hiring to filter smartly. But you have to return something atleast 1 assessment.
# You have to return minimum 1 assessment and maximum 10(only relevant ones). You cannot return empty json.

# Respond in clean JSON format:
# [
#   {{
#     "Assessment Name": "...",
#     "Skills": "...",
#     "Test Type": "...",
#     "Description": "...",
#     "Remote Testing Support": "...",
#     "Adaptive/IRT": "...",
#     "Duration": "... mins",
#     "URL": "...",
#     "Score": ...
#   }},
#   ...
# ]

# ---
# Assessments:
# {top_results}
# """

#     response = gemini_model.generate_content(prompt)
#     return response.text.strip()

# def query_handling_using_LLM_updated(query, model = model , gemini_model = gemini_model, catalog_df = catalog_df, corpus = corpus, corpus_embeddings = corpus_embeddings):
#     url = extract_url_from_text(query)

#     if url:
#         extracted_text = extract_text_from_url(url)
#         query += " " + extracted_text

#     user_query = extract_features_with_llm(query)

#     top_results = find_assessments(user_query, k=10)

#     top_json = json.dumps(top_results, indent=2, default=convert_numpy)

#     filtered_output = filter_relevant_assessments_with_llm(user_query, top_json)

#     # Check for empty response
#     if not filtered_output or not filtered_output.strip():
#         print("Empty response from LLM.")
#         return pd.DataFrame()

#     # Try to extract valid JSON from the output using regex
#     try:
#         match = re.search(r"\[.*\]", filtered_output, re.DOTALL)
#         if match:
#             json_str = match.group()
#             filtered_results = json.loads(json_str)
#         else:
#             print("⚠️ No valid JSON array found in the response:")
#             print(filtered_output)
#             return pd.DataFrame()
#     except json.JSONDecodeError as e:
#         print("JSON Decode Error:", e)
#         print("Raw output was:\n", filtered_output)
#         return pd.DataFrame()

#     # Convert to DataFrame
#     if not filtered_results:
#         return pd.DataFrame()
#     else:
#         try:
#             df = pd.DataFrame(filtered_results)
#             print("Returning DataFrame:\n", df.head())
#             return df
#         except Exception as e:
#             print("Error creating DataFrame:", e)
#             return pd.DataFrame()

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
