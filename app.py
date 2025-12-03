import streamlit as st
import torch
import re
import speech_recognition as sr
from transformers import BertTokenizer, BertForSequenceClassification

# ⚠️ MUST be the very first Streamlit command (after imports)
st.set_page_config(page_title="Emotion Detection App", layout="centered")

# Load tokenizer and model (adjust path as needed)
model_path = "bert_emotion_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

emotion_labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return emotion_labels[predicted_class]

def analyze_file(file):
    content = file.read().decode("utf-8")
    lines = content.strip().splitlines()
    total_chars = len(content)
    special_chars = len(re.findall(r"[^A-Za-z0-9\s]", content))

    st.write("📄 **File Statistics**")
    st.write(f"📝 Total Characters: {total_chars}")
    st.write(f"🔣 Special Characters: {special_chars}")
    st.write(f"📏 Total Lines: {len(lines)}")

    if len(lines) > 0:
        selected_line = st.selectbox("🔍 Choose a line to analyze emotion:", lines)
        if st.button("Analyze Selected Line"):
            emotion = predict_emotion(selected_line)
            st.success(f"❤️ Detected Emotion: **{emotion.upper()}**")
    else:
        st.warning("The file is empty.")

def record_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Please speak.")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.write(f"🗣️ You said: *{text}*")
            emotion = predict_emotion(text)
            st.success(f"❤️ Detected Emotion: **{emotion.upper()}**")
        except sr.UnknownValueError:
            st.error("❌ Could not understand the audio.")
        except sr.RequestError:
            st.error("❌ Request error. Check your internet connection.")
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

# --- Streamlit UI ---

st.title("🧠 Emotion Detection Text Analyzer")
st.markdown("Detect emotions from typed sentences, uploaded `.txt` files, or real-time voice input.")

option = st.radio("Choose input method:", ["Type a Sentence", "Upload .txt File", "Speak (Voice Input)"])

if option == "Type a Sentence":
    user_text = st.text_area("✍️ Enter your sentence:")
    if st.button("Analyze Emotion"):
        if user_text.strip():
            emotion = predict_emotion(user_text)
            st.success(f"❤️ Detected Emotion: **{emotion.upper()}**")
        else:
            st.warning("⚠️ Please enter a valid sentence.")

elif option == "Upload .txt File":
    file = st.file_uploader("📂 Upload a `.txt` file", type=["txt"])
    if file:
        analyze_file(file)

elif option == "Speak (Voice Input)":
    if st.button("🎙️ Start Voice Recording"):
        record_voice()
