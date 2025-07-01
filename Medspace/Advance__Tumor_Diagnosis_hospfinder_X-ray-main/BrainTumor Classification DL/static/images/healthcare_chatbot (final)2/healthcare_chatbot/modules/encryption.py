from cryptography.fernet import Fernet
import os

# Generate a key if not exists
key_file = "encryption_key.key"
if not os.path.exists(key_file):
    with open(key_file, "wb") as f:
        f.write(Fernet.generate_key())

# Load the key
with open(key_file, "rb") as f:
    key = f.read()
cipher = Fernet(key)

# Encrypt text
def encrypt_text(text):
    return cipher.encrypt(text.encode()).decode()

# Decrypt text
def decrypt_text(encrypted_text):
    return cipher.decrypt(encrypted_text.encode()).decode()
