"""
Embedding and Similarity Utilities

Provides functions for generating text embeddings using OpenAI's API
and calculating cosine similarity between vectors.

Example:
    >>> from embeddings import get_embedding, cosine_similarity
    >>> embedding = get_embedding("Hello world")
    >>> similarity = cosine_similarity(vec1, vec2)
"""

import os
import numpy as np
import streamlit as st
from openai import OpenAI
from typing import List, Optional

def _load_api_key() -> str:
    """
    Load OpenAI API key from Streamlit secrets or environment.
    
    Returns:
        API key string
        
    Raises:
        ValueError: If API key not found in either location
    """
    # Try Streamlit secrets first
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Fallback to .env
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. "
                "Please set it in .env file or Streamlit secrets."
            )
        
        return api_key

# Initialize client with validated API key
client = OpenAI(api_key=_load_api_key())


def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Generate embedding vector for text using OpenAI API.
    
    Args:
        text: Text to embed
        model: OpenAI embedding model to use
        
    Returns:
        List of floats representing the embedding vector
        
    Raises:
        RuntimeError: If API call fails
    """
    text = text.replace("\n", " ")
    
    try:
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding: {str(e)}") from e


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
    
    Returns:
        Similarity score between 0 and 1
        
    Raises:
        ValueError: If vectors have zero magnitude
    """
    vec1_arr = np.array(vec1)
    vec2_arr = np.array(vec2)
    
    dot_product = np.dot(vec1_arr, vec2_arr)
    magnitude = np.linalg.norm(vec1_arr) * np.linalg.norm(vec2_arr)
    
    if magnitude == 0:
        raise ValueError("Cannot calculate similarity for zero-length vectors")
    
    return float(dot_product / magnitude)