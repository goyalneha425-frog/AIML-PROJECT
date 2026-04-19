import streamlit as st
import os
import PyPDF2
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. THE LOGIC (The "Brain") ---
def extract_text(file_obj):
    # Streamlit files are "BytesIO" objects, so we handle them like this
    if file_obj.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(file_obj)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    elif file_obj.name.endswith('.docx'):
        return docx2txt.process(file_obj)
    else:
        return str(file_obj.read(), 'utf-8')

def get_matches(jd, resume_list):
    resumes_text = [extract_text(res) for res in resume_list]
    all_content = [jd] + resumes_text
    
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(all_content).toarray()
    
    job_vector = vectors[0].reshape(1, -1)
    resume_vectors = vectors[1:]
    
    similarities = cosine_similarity(job_vector, resume_vectors).flatten()
    
    results = []
    for i, res in enumerate(resume_list):
        results.append({
            "Candidate": res.name, 
            "Score": round(similarities[i] * 100, 2)
        })
    return sorted(results, key=lambda x: x["Score"], reverse=True)

# --- 2. THE UI (The "Face") ---
st.set_page_config(page_title="AI Resume Matcher", layout="wide")

st.title("📄 AI Resume Screening Automation")
st.markdown("Rank resumes based on their relevance to the Job Description using **TF-IDF & Cosine Similarity**.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Job Description")
    jd_input = st.text_area("Paste the JD here...", height=300)

with col2:
    st.subheader("Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("🚀 Rank Resumes"):
    if jd_input and uploaded_files:
        with st.spinner("Analyzing resumes..."):
            rankings = get_matches(jd_input, uploaded_files)
            
            st.success("Analysis Complete!")
            # Display results in a nice table
            st.table(rankings)
    else:
        st.warning("Please provide both a Job Description and at least one Resume.")
