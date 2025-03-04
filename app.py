import streamlit as st
from PyPDF2 import PdfReader
import docx
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text(file, filename):
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file)
    return ""

def extract_skills(text):
    doc = nlp(text.lower())
    return set([token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]])

def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    scores = cosine_similarity([job_vector], resume_vectors).flatten()
    return scores

st.set_page_config(page_title="AI-Resume Ranking System", page_icon="icons8-resume-50.png", layout="wide")

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        * {  /* Apply to ALL elements */
            font-family: "Courier New", Courier, monospace !important;
        }
        body {font-family: 'Cirka', sans-serif}
        h1 {color:rgb(118, 2, 144);}
        h2 {color:rgb(50, 144, 222);}
        h3 {color:rgb(255, 128, 125);}
        .uploadedFile {color:rgb(29, 255, 127); font-weight: bold;}
        .scoreTable {background-color:rgb(255, 185, 185); padding: 10px; border-radius: 10px;}
        .stTextArea textarea {font-size: 14px !important;}
    </style>
""", unsafe_allow_html=True)

st.title("AI Resume Screening & Candidate Ranking System")
st.markdown(" To find the best Candidates seamlessly faster with this AI-Powered Screening and resume checker !!!")

st.header("Job Description ")
job_description = st.text_area("Enter the job description:", height=200)

st.header("Upload Resumes(supports multiple files)")
uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")

    resumes_text = []
    filenames = []

    progress_bar = st.progress(0)
    for idx, file in enumerate(uploaded_files):
        text = extract_text(file, file.name)
        resumes_text.append(text)
        filenames.append(file.name)
        progress_bar.progress((idx + 1) / len(uploaded_files))

    scores = rank_resumes(job_description, resumes_text)
    job_skills = extract_skills(job_description)

    results = []
    for i, text in enumerate(resumes_text):
        resume_skills = extract_skills(text)
        skill_match = len(resume_skills.intersection(job_skills)) / max(len(job_skills), 1)
        results.append({
            "Resume": filenames[i],
            "Score": round(scores[i], 4),
            "Skill Match": round(skill_match, 4),
            "Missing Skills": ", ".join(job_skills - resume_skills) if job_skills - resume_skills else "None"
        })

    results_df = pd.DataFrame(results).sort_values(by="Score", ascending=False)

    st.success("Resume Ranking Complete!")

    st.markdown("### Results Table")
    st.dataframe(results_df.style.format({"Score": "{:.4f}", "Skill Match": "{:.4f}"}))