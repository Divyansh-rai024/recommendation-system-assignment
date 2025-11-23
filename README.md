# Intelligent (SHL) Assessment Recommendation System

This project is an AI-powered recommendation system built to suggest the most relevant SHL assessments based on user queries, job descriptions, or unstructured input data. It combines NLP techniques and Large Language Models (LLMs) to provide accurate, efficient, and context-aware results through an intuitive frontend interface.



## Features

- Recommends SHL assessments based on:
  - Job descriptions
  - Unstructured URLs or text input
  - Custom user queries
- NLP-based semantic similarity matching using Sentence-BERT
- Contextual feature extraction and filtering using Gemini 1.5 Pro (LLM)
- Ranking and scoring using cosine similarity
- Top recommendations output with relevance filtering
- Streamlit-based frontend for an interactive user experience

## Tech Stack

- **Natural Language Processing**:
  - Sentence-BERT for creating embeddings of assessments and queries
  - Cosine similarity for ranking the most relevant assessments
- **LLM Integration**:
  - Gemini 1.5 Pro used to extract structured features (job title, skills, duration, etc.) from unstructured input
  - Post-processing and filtering of recommendations based on constraints like duration and skill match
- **Frontend**:
  - Streamlit application for user-friendly interaction and display of results

## How It Works

1. **Data Preparation**:
   - A mock dataset of 50 SHL-like assessments is used, each containing:
     - Assessment name, URL, duration, test type, skills, description, remote support, and adaptive/IRT support.
   - A "combined" column is created by concatenating all columns into a single string for embedding.

2. **NLP Embedding and Retrieval**:
   - Sentence-BERT is used to convert both dataset entries and input queries into vector embeddings.
   - Cosine similarity is calculated to identify the top matching assessments.

3. **LLM Enhancement (Gemini 1.5 Pro)**:
   - Accepts job descriptions, URLs, or unstructured queries.
   - Extracts meaningful structured features like job role, required skills, expected duration, etc.
   - This information is used to generate more accurate embeddings.
   - After retrieving top candidates, Gemini re-filters based on constraints and relevance.

4. **Evaluation**:
   - Performance is evaluated using metrics such as Recall@5 and MAP@5.
   - The hybrid (NLP + LLM) approach outperforms the pure NLP baseline in both metrics.

5. **Streamlit Interface**:
   - Users can input queries directly in a web interface.
   - Receives and displays the top recommended assessments along with their details.

## Performance
- NLP Model - Recall@5 = 0.85 and MAP@5 = 0.71
- NLP + LLM Model - Recall@5 = 1.0 and MAP@5 = 1.0
  
## Test Cases 
- I am hiring for Java developers who can also collaborate effectively with my business teams. Looking
for an assessment(s) that can be completed in 40 minutes.
- Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an
assessment package that can test all skills with max duration of 60 minutes.
- I am hiring for an analyst and wants applications to screen using Cognitive and personality tests,
what options are available within 45 mins.
- https://www.linkedin.com/jobs/view/research-engineer-ai-at-shl-4194768899/?originalSubdomain=in
- Want to assess communication and teamwork skills in under 30 minutes.
