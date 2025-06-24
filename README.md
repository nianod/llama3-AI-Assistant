# 🧠 Uncensored AI System (LLaMA 3 + CLIP + PyQt5)

A multimodal desktop assistant powered by [Meta LLaMA 3 70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B), OpenAI CLIP, and PyQt5. It supports:

- ✍️ Text generation via LLaMA 3
- 🎙️ Voice input using speech recognition
- 📷 Webcam-based image processing with CLIP
- 🎛️ Optional fine-tuning with PEFT/LoRA
- 💻 GUI interface using PyQt5

---

## ⚙️ Features

| Feature       | Description                                  |
|---------------|----------------------------------------------|
| Text Input    | Enter prompts manually or by microphone      |
| Voice Input   | Uses `speech_recognition` + Google Speech API |
| Webcam Input  | CLIP-based frame processing (if enabled)     |
| Fine-Tuning   | Trigger LoRA fine-tuning inside the app      |

---

## 🚀 Getting Started

###  Clone the Repo and  Run the fillowing commands

```bash 
git clone https://github.com/your-username/llama3-desktop-ai.git
cd llama3-desktop-ai


# 2. Set Up Python Environment
python -m venv venv
source venv/bin/activate  


# 3. Install Dependencies
pip install -r requirements.txt

# 4. Run the app
python llama.py
