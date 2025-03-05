This is an AI-driven web application that automates resume screening and ranking based on job descriptions. 
This system uses Natural Language Processing (NLP) and Machine Learning (ML) to extract skills, compare resumes, and rank candidates based on relevance.
Features
-Upload multiple PDF/DOCX resumes
-Extract and analyze skills from job descriptions and resumes
-Rank candidates using TF-IDF & Cosine Similarity
-Identify missing skills in resumes
-Interactive web app built with Streamlit
Technologies Used
Python (Backend Processing)
Streamlit (Web Application Framework)
spaCy NLP (Skill Extraction)
TF-IDF & Cosine Similarity (Resume Ranking)
PyPDF2 & python-docx (Resume Parsing)
Pandas (Data Handling & Results Processing)
Enter the Job Description
The recruiter enters the job description, which is used to match resumes. 
Upload Resumes
Users can upload multiple resumes in PDF or DOCX format, and the system will extract text from them for processing.
Resume Ranking 
Find all the skill gaps in resumes compared to job descriptions using TF-IDF vectorization and Cosine Similarity. 
View Results
The system displays a ranked list of resumes along with similarity scores and missing skills.
