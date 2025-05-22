from flask import Flask, render_template, request, redirect, url_for, flash, session
from sentence_transformers import SentenceTransformer, util
from werkzeug.utils import secure_filename
import os
import uuid
import PyPDF2
import docx
from extract_skills import extract_skills

app = Flask(__name__)
app.secret_key = 'resume_matcher_secret_key'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(path):
    text = ""
    try:
        with open(path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def extract_text_from_docx(path):
    text = ""
    try:
        doc = docx.Document(path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX: {e}")
    return text

def analyze_resume(job_description, resume_text, resume_name, resume_id):
    job_skills = extract_skills(job_description)
    resume_skills = extract_skills(resume_text)

    job_embed = sbert_model.encode(job_description, convert_to_tensor=True)
    resume_embed = sbert_model.encode(resume_text, convert_to_tensor=True)

    similarity_score = util.pytorch_cos_sim(job_embed, resume_embed).item()
    matching_skills = [skill for skill in resume_skills if skill in job_skills]
    missing_skills = [skill for skill in job_skills if skill not in resume_skills]
    skill_match_score = len(matching_skills) / len(job_skills) if job_skills else 0

    final_score = round(((similarity_score * 60) + (skill_match_score * 40)) * 100, 2)

    return {
        'id': resume_id,
        'name': resume_name,
        'match_score': final_score,
        'matching_skills': matching_skills,
        'missing_skills': missing_skills,
        'content': resume_text[:500] + "..." if len(resume_text) > 500 else resume_text
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'job_description' not in request.form:
        flash('No job description provided', 'danger')
        return redirect(url_for('index'))

    if 'resumes' not in request.files:
        flash('No resume files uploaded', 'danger')
        return redirect(url_for('index'))

    job_description = request.form['job_description']
    session['job_description'] = job_description
    job_skills = extract_skills(job_description)
    session['job_skills'] = job_skills

    batch_id = str(uuid.uuid4())
    session['batch_id'] = batch_id

    files = request.files.getlist('resumes')
    results = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_id = str(uuid.uuid4())
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
            file.save(file_path)

            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif filename.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            else:
                continue

            if text:
                result = analyze_resume(job_description, text, filename, file_id)
                results.append(result)

    session['results'] = results
    session['results_count'] = len(results)

    if results:
        results.sort(key=lambda x: x['match_score'], reverse=True)
        return redirect(url_for('results'))
    else:
        flash('No valid resumes were processed', 'warning')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    if 'results' not in session or not session['results']:
        flash('No results to display.', 'warning')
        return redirect(url_for('index'))
    
    return render_template(
        'results.html',
        results=session['results'],
        results_count=session['results_count'],
        job_skills=session['job_skills']
    )

@app.route('/clear')
def clear():
    session.clear()
    return redirect(url_for('index'))

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Max 16MB.', 'danger')
    return redirect(url_for('index'))

@app.errorhandler(500)
def server_error(e):
    flash('Server error occurred.', 'danger')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
