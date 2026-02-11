"""Embedding generation and similarity calculations."""

import os
import numpy as np
import streamlit as st
from openai import OpenAI


# Try Streamlit secrets first (deployment), fallback to .env (local)
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding vector for text."""
    text = text.replace("\n", " ")

    response = client.embeddings.create(input=[text], model=model)

    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1, vec2: Lists of floats (embedding vectors)
    
    Returns:
        Float between 0 and 1 (similarity score)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    return dot_product / magnitude