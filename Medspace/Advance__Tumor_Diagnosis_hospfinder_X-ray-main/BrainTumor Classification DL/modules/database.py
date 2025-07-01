import os
import pymongo
import bcrypt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["healthcare_db"]
records_collection = db["medical_records"]
users_collection = db["users"]

def register_user(username, password):
    """Registers a new user with hashed password."""
    if users_collection.find_one({"username": username}):
        return False, "Username already exists"
    
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    users_collection.insert_one({"username": username, "password": hashed_password})
    return True, "User registered successfully!"

def authenticate_user(username, password):
    """Authenticates user login."""
    user = users_collection.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        return True
    return False

def save_medical_record(patient_id, text):
    """Saves extracted medical text to MongoDB."""
    record = {"patient_id": patient_id, "text": text}
    records_collection.insert_one(record)

def get_all_records(patient_id):
    """Retrieves all medical records for a specific patient."""
    return list(records_collection.find({"patient_id": patient_id}, {"_id": 0, "text": 1}))
