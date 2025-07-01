import os
import pymongo
import bcrypt
from dotenv import load_dotenv
from modules.encryption import encrypt_text, decrypt_text

# Load environment variables
load_dotenv()

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["healthcare_db"]
users_collection = db["users"]
records_collection = db["medical_records"]

# Register User
def register_user(username, password):
    if users_collection.find_one({"username": username}):
        return {"error": "Username already exists"}

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({"username": username, "password": hashed_password})
    return {"message": "User registered successfully"}

# Authenticate User
def authenticate_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode('utf-8'), user["password"]):
        return True
    return False

# Save Medical Record (Encrypted)
def save_medical_record(patient_id, text):
    encrypted_text = encrypt_text(text)
    records_collection.insert_one({"patient_id": patient_id, "text": encrypted_text})

# Get All Medical Records for Patient (Decrypted)
def get_all_records(patient_id):
    records = records_collection.find({"patient_id": patient_id})
    return [{"text": decrypt_text(record["text"])} for record in records]
