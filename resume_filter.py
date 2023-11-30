from flask import Flask, request, jsonify
import docx2txt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def calculate_similarity(job_description, resume):
    content = [resume, job_description]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(content)
    similarity_matrix = cosine_similarity(count_matrix)
    return similarity_matrix[1][0].round(2) * 100

@app.route('/')
def welcome():
    welcome_message = "Welcome to the Resume and Job Description Similarity Model!"
    instructions = """
    To use this model, send a POST request to /upload with two files:
    1. 'job_description': A .docx file containing the job description.
    2. 'resume': A .docx file containing the resume.
    The model will return the similarity percentage between the job description and the resume.
    """
    return welcome_message + instructions

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'job_description' not in request.files or 'resume' not in request.files:
        return 'Missing files', 400

    job_description_file = request.files['job_description']
    resume_file = request.files['resume']

    job_description = docx2txt.process(job_description_file)
    resume = docx2txt.process(resume_file)

    similarity_score = calculate_similarity(job_description, resume)
    return jsonify({'similarity': similarity_score})

if __name__ == '__main__':
    app.run(debug=True)


