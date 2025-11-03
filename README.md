# SEO Content Quality & Duplicate Detector

## Project Overview

This repository provides a data science pipeline and interactive app for evaluating web page SEO content quality, detecting thin content, and finding duplicate articles. It includes a Jupyter notebook for full reproducibility and a deployed Streamlit app for real-time analysis of any public URL.

## Directory Structure


```bash
seo-content-detector/
├── data/
│   ├── data.csv                 # Provided dataset (URLs + HTML)
│   ├── extracted_content.csv    # Parsed content (no html_content column)
│   ├── features.csv             # Extracted features
│   └── duplicates.csv           # Duplicate pairs
│
├── notebooks/
│   └── seo_pipeline.ipynb       # Main analysis notebook
│
├── streamlit_app/
│   └── app.py                   # Interactive Streamlit app
│
├── models/
│   ├── quality_model.pkl        # Saved classifier
│   └── label_encoder.pkl        # Label encoder (if used)
│
├── requirements.txt
├── .gitignore
└── README.md                    # Documentation



## Setup Instructions

git clone https://github.com/ManavBajaj/seo-content-detector.git
cd seo-content-detector
pip install -r requirements.txt
jupyter notebook notebooks/seo_pipeline.ipynb


## Quick Start

1. Place the provided dataset (`data.csv`) in the `data/` directory.
2. Run `notebooks/seo_pipeline.ipynb` to generate feature, duplicate, and model files.
3. To try the interactive app:
   streamlit run streamlit_app/app.py
4. Analyze any public URL live and export all session results as CSV.

## Deployed Streamlit App

- [App URL](https://manavbajaj-seo-content-detector-streamlit-appapp-fyczzu.streamlit.app/)  
 

## Key Decisions

- Libraries: Chose pandas and BeautifulSoup (with lxml) for robust HTML parsing, textstat for readability, scikit-learn for feature engineering and classification.
- Parsing Approach: Extracted main content using `<main>`, `<article>`, and `<p>` tags; handled malformed HTML gracefully.
- Similarity Threshold: Cosine similarity above 0.80 identified near-duplicates with high reliability.
- Model: Random Forest classifier selected for interpretability and feature ranking; rule-based classifier implemented for task requirements.

## Results Summary

- Model performance: High accuracy and F1 on quality labels; see notebook for metrics and confusion matrix.
- Near-duplicate pages: See results in `data/duplicates.csv` and live via the app.
- Thin content pages: Flagged in `data/features.csv` and session analysis table.
- Typical output: 

| Quality | Word Count | Flesch Score | Thin Content | Duplicates |
|----------|------------|--------------|---------------|-------------|
| High     | 1600       | 65.0         | No            | 0           |
| Low      | 220        | 28.5         | Yes           | 1           |


## Limitations

- Labeling rules are deterministic based on two features; results on subjective or noisy data may vary.
- TF-IDF-based duplicate detection may miss semantic matches not reflected in keyword overlap.
- Extraction may not recover full content on highly non-standard HTML pages.




