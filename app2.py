import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch
from io import StringIO
from PyPDF2 import PdfReader
import docx

# ---------------------------
# App configuration
# ---------------------------
st.set_page_config(
    page_title="Summarizer Chatbot",
    page_icon="📰",
    layout="centered"
)

st.title("🧠 Summarizer Chatbot")
st.caption("Summarize articles and ask questions about the summary using fine-tuned LoRA models (BART / T5).")

# ---------------------------
# Model loader with caching
# ---------------------------
@st.cache_resource
def load_model_and_tokenizer(model_name, lora_path):
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Load both models once
model_bart, tokenizer_bart = load_model_and_tokenizer("facebook/bart-base", "results_bart/checkpoint-1000")
model_t5, tokenizer_t5 = load_model_and_tokenizer("google/flan-t5-base", "results_t5/checkpoint-1000")

# ---------------------------
# Sidebar model selector
# ---------------------------
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.radio(
    "Choose model to use:",
    ("BART", "T5"),
    help="Select which fine-tuned model to use for summarization."
)

# ---------------------------
# Chat-style interface
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None

st.markdown("### 💬 Chat with the Summarizer")

# --- Text input area ---
user_input = st.text_area(
    "Enter your text below:",
    height=200,
    placeholder="Paste a news article or any long text..."
)

# ---------------------------
# Style file uploader like a button
# ---------------------------
st.markdown(
    """
    <style>
    .custom-file-upload > div > label > div {
        background-color: #4CAF50;
        color: white;
        padding: 0.35rem 0.75rem;
        border-radius: 0.25rem;
        text-align: center;
        font-weight: bold;
        cursor: pointer;
        width: 100%;
    }
    .custom-file-upload > div > label > div:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- File uploader + summarize button side by side ---
col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader(
        "", type=["txt", "pdf", "docx"], label_visibility="collapsed", key="custom_file_uploader"
    )
    st.markdown('<div class="custom-file-upload"></div>', unsafe_allow_html=True)
with col2:
    summarize_button = st.button("🪄 Summarize")

# ---------------------------
# Helper function to read uploaded files
# ---------------------------
def read_file_content(uploaded_file):
    if uploaded_file is None:
        return None

    file_type = uploaded_file.name.split(".")[-1].lower()
    text = ""

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
    else:
        st.warning("Unsupported file format.")
        return None

    return text.strip()

# ---------------------------
# Summarization logic
# ---------------------------
def generate_summary(text, model, tokenizer, prefix=""):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = tokenizer(prefix + text, return_tensors="pt", truncation=True, max_length=512).to(device)
    summary_ids = model.generate(**inputs, max_new_tokens=150, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ---------------------------
# Determine text to summarize
# ---------------------------
text_to_summarize = None

if summarize_button and user_input.strip():
    text_to_summarize = user_input
elif uploaded_file:
    text_to_summarize = read_file_content(uploaded_file)
    if text_to_summarize:
        st.success(f"✅ Loaded {uploaded_file.name} successfully!")

# Generate summary if text is available
if text_to_summarize:
    with st.spinner("Generating summary..."):
        if model_choice == "BART":
            summary = generate_summary(text_to_summarize, model_bart, tokenizer_bart)
        else:
            summary = generate_summary("summarize: " + text_to_summarize, model_t5, tokenizer_t5)

    st.session_state.last_summary = summary
    st.session_state.history.append({
        "input": text_to_summarize[:5000],
        "summary": summary,
        "model": model_choice
    })

# ---------------------------
# Display summarization history
# ---------------------------
for chat in reversed(st.session_state.history):
    st.markdown(f"**🗣️ You:** {chat['input'][:1000]}...")
    st.markdown(f"**🤖 {chat['model']} Summary:** {chat['summary']}")
    st.markdown("---")

# ---------------------------
# Q&A Section about the Summary
# ---------------------------
if st.session_state.last_summary:
    st.subheader("🗨️ Ask about the summary or the article")

    question = st.chat_input("Ask a question about the summary...")

    def generate_answer(question, context, model, tokenizer):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        input_text = f"question: {question} context: {context}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(**inputs, max_new_tokens=150, num_beams=4)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    if question:
        with st.spinner("Thinking..."):
            if model_choice == "BART":
                answer = generate_answer(question, st.session_state.last_summary, model_bart, tokenizer_bart)
            else:
                answer = generate_answer(question, st.session_state.last_summary, model_t5, tokenizer_t5)

        st.markdown(f"**🧠 Question:** {question}")
        st.markdown(f"**🤖 Answer:** {answer}")
# --- Add tiny teddy bear emoji in bottom right ---
st.markdown(
    """
    <style>
    .teddy-emoji {
        position: fixed;
        bottom: 20px;
        right: 20px;
        font-size: 30px;  /* size similar to emoji in title */
        z-index: 9999;
    }
    </style>
    <div class="teddy-emoji">🧸</div>
    """,
    unsafe_allow_html=True
)
