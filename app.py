# app.py — Streamlit frontend (NO ML MODELS)
import streamlit as st
import pandas as pd
import requests

# -----------------------------
# CONFIG
# -----------------------------
BACKEND_URL = "https://<YOUR_BACKEND_URL>.onrender.com"   # CHANGE THIS

st.set_page_config(page_title="SHL Assessment Recommendation System", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>🧠 SHL Assessment Recommendation System</h1>
    <h4 style='text-align: center; color: #ccc;'>Find the best assessments based on your query using AI!</h4>
    <hr style="border: 1px solid #333;">
    """,
    unsafe_allow_html=True
)

# -----------------------------
# USER INPUT
# -----------------------------
query = st.text_input("🔍 Enter your search query:", placeholder="e.g. Python SQL coding test")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("🤖 Fetching recommendations from backend..."):
            try:
                res = requests.post(
                    f"{BACKEND_URL}/recommend",
                    json={"query": query},
                    timeout=30
                )

                if res.status_code != 200:
                    st.error(f"Backend error: {res.text}")
                else:
                    data = res.json().get("recommended_assessments", [])
                    if not data:
                        st.warning("😕 No assessments matched your query.")
                    else:
                        df = pd.DataFrame(data)

                        # Make URLs clickable
                        if "url" in df.columns:
                            df["url"] = df["url"].apply(
                                lambda x: f"<a href='{x}' target='_blank'>View</a>" if x else ""
                            )

                        st.success("✅ Top recommended assessments:")

                        st.write(
                            df.to_html(escape=False, index=False),
                            unsafe_allow_html=True
                        )

            except Exception as e:
                st.error(f"🚨 Error communicating with backend: {e}")
