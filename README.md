## 📄 AI-Based Resume Screening and ATS Match Predictor

This project is an **AI-powered Resume Screening System** that evaluates uploaded resumes against selected job roles and calculates an **ATS (Applicant Tracking System) compatibility score**. The system uses natural language processing (NLP) techniques to analyze resume content and match it with predefined job categories, providing insightful feedback and role-specific keyword suggestions.

Users can upload their resumes in PDF, DOCX, or TXT format and select a target job role. The system predicts how well the resume aligns with the selected role using a trained **Naive Bayes classifier** and **TF-IDF** vectorization. It then scales the prediction into a user-friendly score out of 100 and displays match indicators such as ✅ *Excellent Match*, 👍 *Good Match*, ⚠️ *Average Match*, or ❌ *Low Match*.

This tool helps job seekers optimize their resumes for ATS filters and improve their chances of getting shortlisted by tailoring them to specific job roles.

---

## 🛠️ Tools & Technologies Used

### 🧠 Machine Learning & NLP:
- **scikit-learn** – TF-IDF vectorization and Multinomial Naive Bayes classifier
- **pandas** – Data handling and preprocessing
- **joblib** – Model saving and loading

### 📄 Resume Parsing:
- **PyMuPDF (`fitz`)** – Text extraction from PDF files
- **docx2txt** – Text extraction from DOCX files

### 🖥️ Web App & UI:
- **Streamlit** – Fast and interactive frontend framework for Python

---

> 🔍 Designed for educational and demonstration purposes. Can be extended for real-world ATS applications with advanced NLP and resume parsing tools.
