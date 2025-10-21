import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

# ---------------------------
# App configuration
# ---------------------------
st.set_page_config(
    page_title="Summarizer Chatbot",
    page_icon="📰",
    layout="centered"
)

st.title("🧠 Summarizer Chatbot")
st.caption("Choose between fine-tuned LoRA models (BART / T5) to summarize your text.")


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

st.markdown("### 💬 Chat with the Summarizer")
user_input = st.text_area("Enter your text below:", height=200, placeholder="Paste a news article or any long text...")

col1, col2 = st.columns([1, 3])
with col1:
    summarize_button = st.button("🪄 Summarize")

# ---------------------------
# Summarization logic
# ---------------------------
def generate_summary(text, model, tokenizer, prefix=""):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = tokenizer(prefix + text, return_tensors="pt", truncation=True, max_length=512).to(device)
    summary_ids = model.generate(**inputs, max_new_tokens=150, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


if summarize_button and user_input.strip():
    if model_choice == "BART":
        summary = generate_summary(user_input, model_bart, tokenizer_bart)
    else:
        summary = generate_summary("summarize: " + user_input, model_t5, tokenizer_t5)

    st.session_state.history.append({
        "input": user_input,
        "summary": summary,
        "model": model_choice
    })


# ---------------------------
# Display chat history
# ---------------------------
for chat in reversed(st.session_state.history):
    st.markdown(f"**🗣️ You:** {chat['input']}")
    st.markdown(f"**🤖 {chat['model']} Summary:** {chat['summary']}")
    st.markdown("---")
