import os
import secrets
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from modules.database import save_medical_record, get_all_records, register_user, authenticate_user
from modules.chatbot import query_chatbot
from modules.pdf_processor import extract_text_from_pdf

# Load environment variables
load_dotenv()

# Flask App Setup
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(16))  # Secure session management

# File Upload Configuration
UPLOAD_FOLDER = r"C:\Users\Vishnu\Documents\hydfull\Advance_Brain_Tumor_Classification-main\BrainTumor Classification DL\uploadsmr"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {"pdf"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home Page
@app.route('/')
def home6():
    return render_template('indexmr.html')

# User Registration
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username, password = data.get("username"), data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and Password are required"}), 400

    success, message = register_user(username, password)
    return jsonify({"message": message} if success else {"error": message}), (200 if success else 400)

# User Login
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username, password = data.get("username"), data.get("password")

    if authenticate_user(username, password):
        session["user"] = username
        return jsonify({"message": "Login successful!"})
    return jsonify({"error": "Invalid credentials"}), 401

# Logout
@app.route('/logout', methods=['POST'])
def logout():
    session.pop("user", None)
    return jsonify({"message": "Logged out successfully!"})

# Upload Medical Records
@app.route('/upload', methods=['POST'])
def upload_files():
    if "user" not in session:
        return jsonify({"error": "Unauthorized access"}), 403

    if 'files' not in request.files:
        return jsonify({"error": "No files found"}), 400

    files = request.files.getlist('files')
    patient_id = session["user"]  # Use logged-in user as patient ID

    if not files:
        return jsonify({"error": "No files selected"}), 400

    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Extract text and save in DB (Encrypted)
            text = extract_text_from_pdf(file_path)
            save_medical_record(patient_id, text)
            uploaded_files.append(filename)

    return jsonify({"message": "Files uploaded successfully!", "files": uploaded_files})

# Chatbot Query
@app.route('/chat', methods=['POST'])
def chat():
    if "user" not in session:
        return jsonify({"error": "Unauthorized access"}), 403

    data = request.get_json()
    query = data.get("query")

    # Fetch all medical records for the user
    records = get_all_records(session["user"])
    if not records:
        return jsonify({"error": "No medical records found."}), 404

    # Limit the text length to prevent exceeding API limits
    max_chars = 5000
    medical_texts = " ".join(record["text"] for record in records)[:max_chars]
    
    response = query_chatbot(medical_texts, query)
    return jsonify({"response": response})

# Start Flask App
if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
