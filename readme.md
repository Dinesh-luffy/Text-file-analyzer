# 🧠 Emotion Detection Text Analyzer  
A Streamlit-based application that detects human emotions from **text**, **uploaded files**, or **real-time voice input** using a fine-tuned **BERT emotion classification model**.

---

## 🚀 Features  
- **Text Emotion Detection**  
  Enter any sentence and get predicted emotion instantly.

- **Voice Emotion Detection**  
  Speak using your microphone and the app converts voice to text → predicts emotion.

- **Text File Analysis**  
  Upload any `.txt` file to:
  - View total characters  
  - Find number of special characters  
  - Count total lines  
  - Select a line to analyze its emotion

- **GPU Support**  
  Automatically uses CUDA if available for faster inference.

---

## 🎯 Supported Emotions  
Your BERT model predicts one of the following six emotions:

- Anger  
- Fear  
- Joy  
- Love  
- Sadness  
- Surprise  

---

## 🛠️ Tech Stack  
- **Python 3.8+**
- **Streamlit** (frontend UI)
- **PyTorch** (model inference)
- **Transformers (HuggingFace)**  
- **SpeechRecognition + Microphone input**
- **Regular Expressions (for file analysis)**

---

## 📁 Project Structure  

emotion-detection-app/
│── bert_emotion_model/ # Folder containing your fine-tuned BERT model
│── app.py # Your main Streamlit app file
│── requirements.txt
│── README.md


---

## 🔧 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/emotion-detection-app.git
cd emotion-detection-app

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Ensure your model folder exists

Place your model folder:

bert_emotion_model/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.txt

▶️ Run the App
streamlit run app.py


Streamlit will open in your browser at:

http://localhost:8501

🎤 Voice Input Notes

Voice input uses SpeechRecognition and requires:

Internet access (Google Speech API)

A working microphone

If you're in VS Code or PyCharm, ensure microphone permissions are enabled.

📦 Dependencies

All required packages are listed in requirements.txt.

📜 License

This project is open-source and free to use.

👨‍💻 Author

Dinesh Kumar S
AI & ML Enthusiast | Data Science Explorer
