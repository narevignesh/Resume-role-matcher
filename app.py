import streamlit as st
import fitz  # PyMuPDF
import docx2txt
import joblib

# Extract resume text from uploaded file
def extract_text(file):
    if file.type == "application/pdf":
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            return "".join([page.get_text() for page in doc])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    return ""

# Load model once
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

# --- UI START ---
st.set_page_config(page_title="Resume Role Match", page_icon="📄", layout="wide")

st.title("📄 Resume Role Match Predictor")
st.markdown("Upload your resume to see the top matching roles based on its content.")

# Sidebar for resume upload only
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/resume.png", width=70)
    st.header("Upload Resume")
    uploaded_file = st.file_uploader("Choose your resume", type=["pdf", "docx", "txt"])

# Main process
if uploaded_file:
    resume_text = extract_text(uploaded_file)

    if not resume_text.strip():
        st.error("⚠️ Couldn't extract text from the file. Please check the file format or content.")
    else:
        model = load_model()
        predictions = model.predict_proba([resume_text])[0]
        role_score_dict = {role: score * 100 for role, score in zip(model.classes_, predictions)}

        # Sort and get top 2 roles
        top_roles = sorted(role_score_dict.items(), key=lambda x: x[1], reverse=True)[:2]

        # 🎯 Show Top 2 Matching Roles (without score)
        st.subheader("🔍 Top ATS-Matching Roles")
        cols = st.columns(len(top_roles))
        for idx, (role, _) in enumerate(top_roles):
            with cols[idx]:
                st.markdown(
                    f"<div style='font-size:20px; font-weight:bold; color:#1a73e8;'>🟦 {role}</div>", 
                    unsafe_allow_html=True
                )

        # Resume Text Display
        st.divider()
        st.subheader("📄 Extracted Resume Text")
        if st.toggle("Show Copyable Resume Text", value=True):
            st.code(resume_text, language="text")
        else:
            st.text_area("Preview Resume Text", resume_text, height=300)

else:
    st.info("📌 Please upload your resume to see the top matching roles.")
