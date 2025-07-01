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
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(16))  # Session security

# Uploads Folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# User Registration
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username, password = data.get("username"), data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and Password are required"}), 400

    result = register_user(username, password)
    return jsonify(result)

# User Login
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username, password = data.get("username"), data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and Password are required"}), 400

    if authenticate_user(username, password):
        session["username"] = username
        return jsonify({"message": "Login successful"})
    else:
        return jsonify({"error": "Invalid credentials"}), 401

# Logout
@app.route('/logout', methods=['POST'])
def logout():
    session.pop("username", None)
    return jsonify({"message": "Logged out successfully"})

# Upload Medical Records
@app.route('/upload', methods=['POST'])
def upload_files():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 403

    if "files" not in request.files:
        return jsonify({"error": "No file part"}), 400

    patient_id = session["username"]  # Using username as patient ID
    files = request.files.getlist("files")

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Extract text and save in DB (Encrypted)
            text = extract_text_from_pdf(file_path)
            save_medical_record(patient_id, text)

    return jsonify({"message": "Files uploaded successfully!"})

# Chatbot Query
@app.route('/chat', methods=['POST'])
def chat():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json()
    query = data.get("query")

    # Fetch all medical records for the patient
    records = get_all_records(session["username"])
    if not records:
        return jsonify({"error": "No medical records found."}), 404

    medical_texts = " ".join(record["text"] for record in records)
    response = query_chatbot(medical_texts, query)

    return jsonify({"response": response})

# Start Flask App
if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5001)))
