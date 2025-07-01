import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Groq API Key Setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

def query_chatbot(medical_text, user_query):
    """Sends all patient records and query to the AI model."""
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant analyzing multiple patient reports."},
            {"role": "user", "content": f"Patient's medical history:\n{medical_text}\n\nUser query: {user_query}"}
        ]
    )
    return response.choices[0].message.content
