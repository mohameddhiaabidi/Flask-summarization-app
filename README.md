# Summarization Flask App

This app accepts PDF uploads in English or French and summarizes them using pretrained models.

## Features
- Language detection
- Semantic chunking
- Model-based summarization (English: LaMini-Flan-T5, French: mBART)
- Self-similarity coherence score

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python summAPP.py

## http://localhost:5000/summarize