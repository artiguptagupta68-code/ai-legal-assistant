# ==========================================
# 📄 AI LEGAL DOCUMENT ASSISTANT (FINAL)
# ==========================================

import streamlit as st
import spacy
from transformers import pipeline

# -------------------------------
# 🔹 Load Models (Optimized)
# -------------------------------
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")

    # ✅ Lightweight summarizer
    summarizer = pipeline(
        task="summarization",
        model="sshleifer/distilbart-cnn-6-6"
    )

    # ✅ Lightweight smart AI model
    generator = pipeline(
        task="text2text-generation",
        model="google/flan-t5-small"
    )

    return nlp, summarizer, generator


nlp, summarizer, generator = load_models()

# -------------------------------
# 🔹 Functions
# -------------------------------
def generate_summary(text):
    try:
        if len(text) < 50:
            return "⚠️ Text too short to summarize"
        result = summarizer(text, max_length=80, min_length=25, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"⚠️ Summary error: {str(e)}"


def extract_entities(text):
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text)
    return entities


def extract_clauses(text):
    text_lower = text.lower()
    clauses = {}

    if "terminate" in text_lower:
        clauses["Termination Clause"] = "Termination condition present"

    if "payment" in text_lower or "amount" in text_lower:
        clauses["Payment Clause"] = "Payment terms mentioned"

    if "confidentiality" in text_lower:
        clauses["Confidentiality Clause"] = "Confidentiality clause present"

    if "notice" in text_lower:
        clauses["Notice Period"] = "Notice period mentioned"

    return clauses


def detect_risks(text):
    text_lower = text.lower()
    risks = []

    if "dispute" not in text_lower:
        risks.append("⚠️ Missing dispute resolution clause")

    if "penalty" not in text_lower:
        risks.append("⚠️ No penalty clause")

    if "liability" not in text_lower:
        risks.append("⚠️ Liability clause missing")

    return risks


def ask_question(question, context):
    try:
        prompt = f"""
You are a legal assistant. Answer clearly based only on the document.

Document:
{context}

Question:
{question}

Answer:
"""
        result = generator(prompt, max_length=150, do_sample=False)
        answer = result[0]["generated_text"]

        if "Answer:" in answer:
            return answer.split("Answer:")[-1].strip()
        return answer.strip()

    except Exception as e:
        return f"⚠️ Error generating answer: {str(e)}"


# -------------------------------
# 🔹 UI Layout
# -------------------------------
st.set_page_config(page_title="AI Legal Assistant", layout="wide")

st.title("📄 AI Legal Document Assistant")
st.markdown("Analyze legal documents using AI (Summary, Clauses, Risks, Q&A)")

# Input
text_input = st.text_area("📥 Paste Legal Document Here", height=250)

# -------------------------------
# 🔹 Analyze Button
# -------------------------------
if st.button("🔍 Analyze Document"):

    if not text_input.strip():
        st.warning("Please enter legal text")
    else:
        # Summary
        st.subheader("🧾 Summary")
        st.success(generate_summary(text_input))

        # Entities
        st.subheader("🏷️ Entities")
        st.json(extract_entities(text_input))

        # Clauses
        st.subheader("🔑 Key Clauses")
        clauses = extract_clauses(text_input)
        if clauses:
            st.write(clauses)
        else:
            st.warning("No major clauses detected")

        # Risks
        st.subheader("⚠️ Risk Analysis")
        risks = detect_risks(text_input)
        if risks:
            for r in risks:
                st.error(r)
        else:
            st.success("No major risks detected")

# -------------------------------
# 🔹 Smart Q&A Section
# -------------------------------
st.subheader("💬 Ask Questions (Smart AI)")

question = st.text_input("Ask anything about the document")

if question and text_input:
    answer = ask_question(question, text_input)

    if not answer.strip():
        st.warning("No clear answer found")
    else:
        st.success(answer)
