import os
from flask import Flask, render_template, request, jsonify
from Core.matcher import final_Result

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')




@app.route('/predict', methods=['POST'])
def predict():
    
    if 'cv' not in request.files or 'jd' not in request.form:
        return jsonify({"error": "Missing CV file or Job Description."}), 400

    cv_file = request.files['cv']
    jd_text = request.form['jd']

    if cv_file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    # Save file temporarily
    temp_cv_path = os.path.join(UPLOAD_FOLDER, cv_file.filename)
    cv_file.save(temp_cv_path)

    try:
        
        result = final_Result(temp_cv_path, jd_text)
    finally:
        # Clean up file from disk
        if os.path.exists(temp_cv_path):
            os.remove(temp_cv_path)

    
    return render_template('result.html', results=result)

if __name__ == '__main__':
    app.run(debug=True)