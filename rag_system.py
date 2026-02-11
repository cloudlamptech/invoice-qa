import os
import glob
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from pdf_reader import extract_text_from_pdf, chunk_text
from embeddings import get_embedding, cosine_similarity
from query import answer_question


def main():
    # Step 1: Extract PDFs
    pdf_files = glob.glob('data/0008998092500017_ebcd3960-e769-482c-a491-8297278c2e96.pdf')
    print(f"Found {len(pdf_files)} PDF files")

    # Extract all documents
    documents = []
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if text:
            documents.append({
                'filename': pdf_file,
                'content': text
            })
            print(f"✓ Extracted {len(text)} characters from {pdf_file}")

    print(f"\nTotal documents loaded: {len(documents)}")

    # Chunk the documents
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc['content'], chunk_size=500, overlap=50)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'text': chunk,
                'source': doc['filename'],
                'chunk_id': i
            })

    # Generate Embeddings
    print("\n" + "="*60)
    print("GENERATING EMBEDDINGS (Testing)")
    print("="*60)

    for i, chunk in enumerate(all_chunks):
        print(f"\nChunk {i+1}:")
        print(f"Text preview: {chunk['text'][:100]}...")

        #Generate Embedding
        embedding = get_embedding(chunk['text'])
        chunk['embedding'] = embedding

    print("\n" + "="*60)
    print(f"✓ ALL DONE! Generated {len(all_chunks)} embeddings")
    print("="*60)

    questions = [
        "How much CGST did I pay?",
        # "What is the total invoice amount?",
        "What items did I order?",
        # "Who is the invoice issued to?",
        # "What is the restaurant name?",
        "What is the total GST I pay (all taxes)?"
    ]

    for question in questions:
        answer = answer_question(question, all_chunks, top_k=3)
        
        # Pause between questions (optional)
        if question != questions[-1]:  # Don't pause after last question
            input("\nPress Enter for next question...")

if __name__ == "__main__":
    main()