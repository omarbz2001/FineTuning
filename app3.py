import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline
from peft import PeftModel
import torch
from io import StringIO
from PyPDF2 import PdfReader
import docx
import pandas as pd

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="AI Summarizer, Sentiment & Q&A", page_icon="🧠", layout="centered")

st.title("🧠 AI Summarizer, Sentiment & Emotion Chatbot")
st.caption("Summarize any article, analyze its tone, and ask questions about it — powered by fine-tuned T5/BART models.")

# ---------------------------
# Load Models
# ---------------------------
@st.cache_resource
def load_summarizer_models():
    model_bart = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
    tokenizer_bart = AutoTokenizer.from_pretrained("facebook/bart-base")
    model_t5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    tokenizer_t5 = AutoTokenizer.from_pretrained("google/flan-t5-base")
    return model_bart, tokenizer_bart, model_t5, tokenizer_t5

model_bart, tokenizer_bart, model_t5, tokenizer_t5 = load_summarizer_models()

# Sentiment + Emotion Models
@st.cache_resource
def load_analysis_models():
    sentiment_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    emotion_pipe = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)
    return sentiment_pipe, emotion_pipe

sentiment_pipe, emotion_pipe = load_analysis_models()

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.radio("Choose summarization model:", ("BART", "T5"))

# ---------------------------
# File or Text Input
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None

user_input = st.text_area("Paste your article or text here:", height=200)

col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader("", type=["txt", "pdf", "docx"], label_visibility="collapsed")
with col2:
    summarize_button = st.button("🪄 Summarize")

# ---------------------------
# Helper Functions
# ---------------------------
def read_file_content(uploaded_file):
    if uploaded_file is None:
        return None
    text = ""
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type == "txt":
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        text = stringio.read()
    elif file_type == "pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif file_type == "docx":
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text.strip()

def generate_summary(text, model, tokenizer, prefix=""):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = tokenizer(prefix + text, return_tensors="pt", truncation=True, max_length=512).to(device)
    summary_ids = model.generate(**inputs, max_new_tokens=150, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def analyze_sentiment(text):
    result = sentiment_pipe(text)[0]
    return result["label"], round(result["score"], 3)

def analyze_emotion(text):
    results = emotion_pipe(text)[0]
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    top_emotion = results[0]
    df = pd.DataFrame(results)
    return top_emotion["label"], round(top_emotion["score"], 3), df

def generate_answer(question, context, model, tokenizer):
    """Q&A generation using the summarization model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_new_tokens=150, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------
# Run Summarization + Analysis
# ---------------------------
text_to_summarize = None
if summarize_button and user_input.strip():
    text_to_summarize = user_input
elif uploaded_file:
    text_to_summarize = read_file_content(uploaded_file)
    if text_to_summarize:
        st.success(f"✅ Loaded {uploaded_file.name} successfully!")

if text_to_summarize:
    with st.spinner("Generating summary..."):
        if model_choice == "BART":
            summary = generate_summary(text_to_summarize, model_bart, tokenizer_bart)
        else:
            summary = generate_summary("summarize: " + text_to_summarize, model_t5, tokenizer_t5)

    # 🔍 Sentiment & Emotion
    with st.spinner("Analyzing sentiment and emotions..."):
        sentiment_label, sentiment_score = analyze_sentiment(summary)
        emotion_label, emotion_score, emotion_df = analyze_emotion(summary)

    st.session_state.last_summary = summary

    # Store in session
    st.session_state.history.append({
        "input": text_to_summarize[:1000],
        "summary": summary,
        "sentiment": sentiment_label,
        "sentiment_score": sentiment_score,
        "emotion": emotion_label,
        "emotion_score": emotion_score,
        "emotion_df": emotion_df,
        "model": model_choice
    })

# ---------------------------
# Display Results
# ---------------------------
for chat in reversed(st.session_state.history):
    st.markdown(f"### 🧾 Original Text")
    st.write(chat['input'][:500] + "...")
    st.markdown(f"### 🧠 Summary ({chat['model']})")
    st.success(chat['summary'])

    st.markdown(f"### 💬 Sentiment")
    st.info(f"**{chat['sentiment']}** (confidence: {chat['sentiment_score']})")

    st.markdown(f"### 🎭 Emotion Analysis")
    st.write(f"**Dominant Emotion:** {chat['emotion']} ({chat['emotion_score']})")
    st.bar_chart(chat["emotion_df"].set_index("label")["score"])

    st.markdown("---")

# ---------------------------
# 🧠 Q&A System
# ---------------------------
if st.session_state.last_summary:
    st.subheader("🗨️ Ask Questions about the Summary")
    question = st.chat_input("Type your question here...")

    if question:
        with st.spinner("Thinking..."):
            if model_choice == "BART":
                answer = generate_answer(question, st.session_state.last_summary, model_bart, tokenizer_bart)
            else:
                answer = generate_answer(question, st.session_state.last_summary, model_t5, tokenizer_t5)

        st.markdown(f"**🧠 Question:** {question}")
        st.markdown(f"**🤖 Answer:** {answer}")
        st.markdown("---")

# 🧸 Decorative Teddy Bear
st.markdown("""
<style>
.teddy-emoji {
    position: fixed;
    bottom: 20px;
    right: 20px;
    font-size: 30px;
    z-index: 9999;
}
</style>
<div class="teddy-emoji">🧸</div>
""", unsafe_allow_html=True)
