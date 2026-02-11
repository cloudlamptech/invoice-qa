"""Question answering with RAG."""

import os
from openai import OpenAI
import streamlit as st
from embeddings import get_embedding, cosine_similarity


# Try Streamlit secrets first (deployment), fallback to .env (local)
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

def answer_question(question, all_chunks, top_k=3):
    """
    Answer a question using RAG.
    
    Args:
        question: User's question (string)
        all_chunks: List of chunks with embeddings
        top_k: Number of chunks to retrieve
    
    Returns:
        String answer from LLM
    """
    
    # STEP 1: Convert question to embedding
    question_embedding = get_embedding(question)
    
    # STEP 2: Calculate similarity with all chunks
    similarities = []
    for chunk in all_chunks:
        score = cosine_similarity(question_embedding, chunk['embedding'])
        similarities.append({
            'chunk': chunk,
            'score': score
        })
    
    # STEP 3: Sort and get top K chunks
    similarities.sort(key=lambda x: x['score'], reverse=True)
    top_chunks = similarities[:top_k]
    
    # Combine top chunks into context
    context = "\n\n---\n\n".join([item['chunk']['text'] for item in top_chunks])
    
    # Create the prompt
    #prompt = f"""Answer the question based ONLY on the context below. If the answer is not in the context, say "I don't know based on the provided documents."
    prompt = f"""Answer the question using the context below. If the context doesn't explicitly contain the answer but you can reasonably infer it from the information provided (like determining if a food item is vegetarian based on its name), provide the answer and note that it's an inference.

Context:
{context}

Question: {question}

Answer:"""
    
    # Call OpenAI Chat API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided context. Be concise and accurate."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    answer = response.choices[0].message.content
    
    return answer