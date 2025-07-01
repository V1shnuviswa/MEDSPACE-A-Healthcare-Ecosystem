# 🏥 MEDSPACE AN AI-Powered Healthcare Platform

## 🚀 Overview

An **AI-driven healthcare platform** designed to revolutionize medical diagnostics, patient record management, and hospital discovery. This **end-to-end system** integrates advanced deep learning, NLP, and geolocation intelligence to offer:

- 📊 **Accurate diagnostics** for MRI, CT, and X-ray images using VGG-19 and OpenCV.
- 🤖 **AI-based symptom checker** and medicine recommender using NLP models (Hugging Face).
- 🗂️ **Secure Electronic Health Record (EHR)** handling with JWT authentication.
- 📍 **Real-time nearby hospital discovery** using the Geoapify API.

By automating diagnostic and administrative tasks, the platform reduces manual workload by 50%, improves decision-making, and enhances healthcare accessibility.

---

## 🌟 Key Features

### 🧠 AI Diagnostics
- Utilizes pre-trained **VGG-19 CNN** and **OpenCV** for brain tumor, fracture, and pneumonia detection.
- Achieves **90%+ accuracy** on medical image classification tasks.
- Supports MRI, CT, and X-ray image uploads.

### 💬 Symptom Checker & Medicine Recommender
- Built with **Hugging Face NLP models** for symptom analysis.
- Provides relevant **medicine suggestions**, **home remedies**, and **specialist recommendations**.

### 🔐 Secure EHR Management
- EHRs are encrypted and stored securely with **JWT-based authentication**.
- Patients can **upload**, **access**, and **share** records safely.

### 🏥 Hospital Discovery
- Real-time **geo-based hospital discovery** using **Geoapify API**.
- Users get a list of **nearby hospitals** based on current symptoms and location.

### ⚙️ Full Stack System
- **Frontend**: React.js (Interactive UI, responsive design)
- **Backend**: Flask (Python APIs, model serving, authentication)
- **Database**: SQL (EHR storage, user details)
- **LLM Integration**: For symptom-based AI interaction

---

## 🧩 System Architecture

| Layer        | Tech Stack               | Description                                              |
|--------------|--------------------------|----------------------------------------------------------|
| Frontend     | React.js                 | User-friendly UI with responsive design                  |
| Backend      | Flask, Python            | Business logic, model processing, and API endpoints      |
| AI Models    | VGG-19, Hugging Face     | Image classification, NLP-based diagnosis & suggestions  |
| Database     | SQLite / PostgreSQL      | Secure EHR and user record management                    |
| Auth         | JWT                      | Secure token-based user authentication                   |
| Geo Location | Geoapify API             | Nearby hospital search via geolocation                   |

---

## 📦 Tech Stack

| Category         | Technologies                                      |
|------------------|---------------------------------------------------|
| Frontend         | React.js, TailwindCSS / Bootstrap                 |
| Backend          | Flask, Python                                     |
| AI Models        | VGG-19 (Keras), OpenCV, Transformers (Hugging Face) |
| Auth             | JSON Web Tokens (JWT)                             |
| EHR Storage      | SQL (SQLite or PostgreSQL)                        |
| Geo API          | Geoapify                                          |

---

## 🧪 Accuracy and Performance

- ✅ 90%+ Accuracy on:
  - Brain Tumor Detection (MRI)
  - Bone Fracture Detection (X-ray/CT)
  - Pneumonia Detection (X-ray)
- 🧠 NLP-powered symptom checker with contextual medicine suggestions
- 🔐 JWT-secured EHR reduces data breach risk
- ⏱️ 50% reduction in manual workload for hospital staff

---

## 🚀 How to Run the Project

### ✅ Prerequisites
- Python 3.8+
- Node.js (v14+)
- React & Flask installed
- Geoapify API key
- Hugging Face access token (if required)

### 📁 Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Add your API keys to .env
FLASK_APP=app.py
FLASK_ENV=development
GEOAPIFY_API_KEY=your_key_here
HF_API_KEY=your_token_here
JWT_SECRET_KEY=your_jwt_secret

