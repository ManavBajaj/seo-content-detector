import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import joblib
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load features and models
features_db = pd.read_csv("../data/features.csv")
existing_embeddings = np.array(features_db['embedding'].apply(eval).tolist())
existing_urls = features_db['url'].tolist()
quality_model = joblib.load("../models/quality_model.pkl")
label_encoder = joblib.load("../models/label_encoder.pkl")
corpus = features_db['top_keywords'].str.replace('|', ' ').tolist()
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
vectorizer.fit(corpus)

def get_top_keywords_from_vec(row_vec, feature_names, top_k=5):
    idxs = np.argsort(row_vec)[::-1][:top_k]
    return "|".join(feature_names[idxs])

def analyze_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        return {"error": f"Failed to fetch page: {e}"}
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    main_content = ""
    for tag in ["main", "article", "body"]:
        found = soup.find(tag)
        if found:
            main_content = found.get_text(separator=" ", strip=True)
            if len(main_content.split()) > 20:
                break
    if not main_content or len(main_content.split()) < 20:
        paragraphs = soup.find_all("p")
        main_content = " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs])
    body_text = re.sub(r'\s+', ' ', main_content).strip()
    body_clean = body_text.lower().strip()
    word_count = len(body_clean.split())
    sentence_count = textstat.sentence_count(body_clean)
    readability = textstat.flesch_reading_ease(body_clean)
    new_corpus = corpus + [body_clean]
    vectorizer_fit = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer_fit.fit_transform(new_corpus)
    new_embedding = tfidf_matrix[-1].toarray().ravel()
    feature_names = np.array(vectorizer_fit.get_feature_names_out())
    top_keywords = get_top_keywords_from_vec(new_embedding, feature_names)
    model_features = np.array([[word_count, sentence_count, readability]])
    label_idx = quality_model.predict(model_features)[0]
    quality_label = label_encoder.inverse_transform([label_idx])[0]
    is_thin = word_count < 500
    sim_scores = cosine_similarity([new_embedding], existing_embeddings).flatten()
    similar_to = []
    for idx, sim in enumerate(sim_scores):
        if sim > 0.80:
            similar_to.append({"url": existing_urls[idx], "similarity": float(f"{sim:.2f}")})
    result = {
        "url": url,
        "title": title,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "readability": float(f"{readability:.2f}"),
        "quality_label": quality_label,
        "is_thin": is_thin,
        "top_keywords": top_keywords,
        "similar_to": similar_to
    }
    return result

# ------ STREAMLIT UI --------
st.title("SEO Content Quality & Duplicate Detector")
user_url = st.text_input("Enter a URL to analyze:", "")
if st.button("Analyze") and user_url.strip():
    with st.spinner('Analyzing...'):
        output = analyze_url(user_url.strip())
    if "error" in output:
        st.error(output["error"])
    else:
        st.success(f"Analysis for: {output['url']}")
        st.write(f"**Title:** {output['title']}")
        st.write(f"**Word count:** {output['word_count']}")
        st.write(f"**Sentence count:** {output['sentence_count']}")
        st.write(f"**Flesch Reading Ease:** {output['readability']}")
        st.write(f"**Quality Label:** :blue[{output['quality_label'].upper()}]")
        st.write(f"**Is Thin Content?** {output['is_thin']}")
        st.write(f"**Top 5 Keywords:** {output['top_keywords'].replace('|', ', ')}")
        if output['similar_to']:
            st.write("### Near-Duplicate Pages:")
            for dup in output['similar_to']:
                st.write(f"- {dup['url']} (similarity: {dup['similarity']})")
        else:
            st.write("No near-duplicates found.")

st.markdown("---")
st.caption("Assignment Demo – Data Science SEO Content Detector")

