import streamlit as st
import os
from pdf_reader import extract_text_from_pdf, chunk_text
from embeddings import get_embedding
from query import answer_question

st.set_page_config(page_title="Invoice QA Assistant", page_icon="📄")

# ============================================================
# PRODUCTION SAFEGUARDS
# ============================================================

# Rate limiting using session state
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'uploaded_count' not in st.session_state:
    st.session_state.uploaded_count = 0

# Limits
MAX_QUERIES_PER_SESSION = 10  # Free tier limit
MAX_FILES_PER_UPLOAD = 3      # Prevent abuse
MAX_FILE_SIZE_MB = 5          # MB per file
MAX_CHUNKS_TOTAL = 50         # Prevent huge bills

# ============================================================
# TITLE & INTRO
# ============================================================

st.title("📄 Invoice Question Answering")
st.write("Upload your invoice PDFs and ask questions using AI!")

# Show usage limits
col1, col2 = st.columns(2)
with col1:
    st.metric("Questions Asked", f"{st.session_state.query_count}/{MAX_QUERIES_PER_SESSION}")
with col2:
    st.metric("Files Uploaded", f"{st.session_state.uploaded_count}/{MAX_FILES_PER_UPLOAD}")

# ============================================================
# SIDEBAR - FILE UPLOAD
# ============================================================

with st.sidebar:
    st.header("Upload Invoices")
    
    # Check upload limit
    if st.session_state.uploaded_count >= MAX_FILES_PER_UPLOAD:
        st.error(f"⚠️ Upload limit reached ({MAX_FILES_PER_UPLOAD} files per session). Refresh page to reset.")
        uploaded_files = None
    else:
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type=['pdf'], 
            accept_multiple_files=True,
            help=f"Max {MAX_FILES_PER_UPLOAD} files, {MAX_FILE_SIZE_MB}MB each"
        )
    
    if st.button("Process Documents", disabled=st.session_state.uploaded_count >= MAX_FILES_PER_UPLOAD):
        if uploaded_files:
            # Validation 1: Check file count
            if len(uploaded_files) > MAX_FILES_PER_UPLOAD:
                st.error(f"❌ Too many files! Maximum {MAX_FILES_PER_UPLOAD} files allowed.")
            else:
                # Validation 2: Check file sizes
                oversized_files = []
                for f in uploaded_files:
                    size_mb = f.size / (1024 * 1024)
                    if size_mb > MAX_FILE_SIZE_MB:
                        oversized_files.append(f"{f.name} ({size_mb:.1f}MB)")
                
                if oversized_files:
                    st.error(f"❌ Files too large (max {MAX_FILE_SIZE_MB}MB):\n" + "\n".join(oversized_files))
                else:
                    with st.spinner("Processing PDFs..."):
                        all_chunks = []
                        
                        for uploaded_file in uploaded_files:
                            # Save to temp location
                            temp_path = f"temp_{uploaded_file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Process
                            text = extract_text_from_pdf(temp_path)
                            
                            # Validation 3: Check if PDF has text
                            if not text or len(text.strip()) < 50:
                                st.warning(f"⚠️ {uploaded_file.name} appears to be empty or image-based (no text extracted)")
                                os.remove(temp_path)
                                continue
                            
                            chunks = chunk_text(text)
                            
                            for i, chunk in enumerate(chunks):
                                all_chunks.append({
                                    'text': chunk,
                                    'source': uploaded_file.name,
                                    'chunk_id': i
                                })
                            
                            # Clean up temp file
                            os.remove(temp_path)
                        
                        # Validation 4: Check total chunks
                        if len(all_chunks) > MAX_CHUNKS_TOTAL:
                            st.error(f"❌ Too much content! Generated {len(all_chunks)} chunks (max {MAX_CHUNKS_TOTAL}). Try fewer or smaller files.")
                        elif len(all_chunks) == 0:
                            st.error("❌ No valid content extracted from PDFs.")
                        else:
                            # Generate embeddings with cost estimate
                            tokens_estimate = sum(len(chunk['text'].split()) for chunk in all_chunks) * 1.3
                            cost_estimate = (tokens_estimate / 1_000_000) * 0.02  # $0.02 per 1M tokens
                            
                            st.info(f"📊 Generating embeddings for {len(all_chunks)} chunks (~${cost_estimate:.4f})")
                            
                            for i, chunk in enumerate(all_chunks):
                                chunk['embedding'] = get_embedding(chunk['text'])
                                if (i + 1) % 10 == 0:
                                    st.write(f"  ✓ {i + 1}/{len(all_chunks)} embeddings")
                            
                            # Store in session state
                            st.session_state.chunks = all_chunks
                            st.session_state.uploaded_count += len(uploaded_files)
                            st.success(f"✓ Processed {len(uploaded_files)} file(s), created {len(all_chunks)} chunks")

# ============================================================
# MAIN AREA - QUESTION ANSWERING
# ============================================================

st.header("Ask Questions")

if 'chunks' not in st.session_state:
    st.info("👈 Upload and process invoices first")
else:
    st.write(f"📊 Ready! {len(st.session_state.chunks)} chunks loaded from {st.session_state.uploaded_count} file(s)")
    
    # Question input
    question = st.text_input(
        "Your question:", 
        placeholder="How much CGST did I pay?",
        max_chars=200  # Validation 5: Limit question length
    )
    
    # Check query limit
    if st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
        st.error(f"⚠️ Query limit reached ({MAX_QUERIES_PER_SESSION} per session). Refresh page to reset.")
        answer_disabled = True
    else:
        answer_disabled = False
    
    if st.button("Get Answer", disabled=answer_disabled) and question:
        if len(question.strip()) < 5:
            st.warning("⚠️ Please ask a more detailed question (min 5 characters)")
        else:
            with st.spinner("Finding answer..."):
                # Increment counter
                st.session_state.query_count += 1
                
                # Cost estimate
                context_tokens = sum(len(chunk['text'].split()) for chunk in st.session_state.chunks[:3]) * 1.3
                llm_cost = ((context_tokens + 100) / 1_000_000) * 0.150 + (200 / 1_000_000) * 0.600
                
                # Get answer
                answer = answer_question(question, st.session_state.chunks, top_k=3)
                
                st.success(f"✓ {answer}")
                
                # Show cost (for transparency)
                with st.expander("ℹ️ Query Details"):
                    st.caption(f"Estimated cost: ~${llm_cost:.4f}")
                    st.caption(f"Queries remaining: {MAX_QUERIES_PER_SESSION - st.session_state.query_count}")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("Built with OpenAI GPT-4o-mini and Embeddings API • Free tier: 10 queries/session")

# Reset button (for testing)
if st.sidebar.button("🔄 Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()