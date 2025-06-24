import sys
import threading
import speech_recognition as sr
import cv2
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QPushButton, 
                           QVBoxLayout, QWidget, QLabel, QHBoxLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from transformers import (AutoModelForCausalLM, AutoTokenizer, CLIPProcessor, 
                        CLIPModel, TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json

# ======================
# CONFIGURATION
# ======================

MODEL_NAME = "meta-llama/Meta-Llama-3-70B"  
VOICE_ENABLED = True
WEBCAM_ENABLED = True
FINE_TUNE_ENABLED = False 

# ======================
# AI CORE SYSTEM
# ======================
class UncensoredAI:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.clip_model = None
        self.clip_processor = None
        self.voice_recognizer = sr.Recognizer()
        self.load_models()
        
    def load_models(self):
        # Load Llama 3 70B
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load CLIP for webcam
        if WEBCAM_ENABLED:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    def generate_text(self, prompt, max_length=4096, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def process_voice(self):
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.voice_recognizer.listen(source)
            try:
                return self.voice_recognizer.recognize_google(audio)
            except:
                return ""
    
    def process_webcam(self, frame):
        inputs = self.clip_processor(images=frame, return_tensors="pt", padding=True)
        outputs = self.clip_model(**inputs)
        return "Webcam analysis complete" 

# ======================
# GUI INTERFACE
# ======================
class AIWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ai = UncensoredAI()
        self.init_ui()
        if WEBCAM_ENABLED:
            self.init_webcam()
        
    def init_ui(self):
        self.setWindowTitle("Uncensored AI System (Llama 3 70B)")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widgets
        self.text_input = QTextEdit()
        self.text_output = QTextEdit()
        self.webcam_label = QLabel()
        self.webcam_label.setFixedSize(640, 480)
        
        # Buttons
        self.btn_generate = QPushButton("Generate Text (Ctrl+Enter)")
        self.btn_voice = QPushButton("Voice Input")
        self.btn_finetune = QPushButton("Start Fine-Tuning")
        
        # Layout
        main_layout = QHBoxLayout()
        
        left_panel = QVBoxLayout()
        left_panel.addWidget(self.text_input)
        left_panel.addWidget(self.btn_generate)
        left_panel.addWidget(self.btn_voice)
        if FINE_TUNE_ENABLED:
            left_panel.addWidget(self.btn_finetune)
        
        right_panel = QVBoxLayout()
        right_panel.addWidget(self.text_output)
        if WEBCAM_ENABLED:
            right_panel.addWidget(self.webcam_label)
        
        main_layout.addLayout(left_panel, 60)
        main_layout.addLayout(right_panel, 40)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # Connections
        self.btn_generate.clicked.connect(self.on_generate)
        self.btn_voice.clicked.connect(self.on_voice_input)
        if FINE_TUNE_ENABLED:
            self.btn_finetune.clicked.connect(self.start_fine_tuning)
    
    def init_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_webcam)
        self.timer.start(30)
    
    def update_webcam(self):
        ret, frame = self.capture.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
            self.webcam_label.setPixmap(QPixmap.fromImage(p))
    
    def on_generate(self):
        prompt = self.text_input.toPlainText()
        if not prompt:
            return
        
        def generate_thread():
            response = self.ai.generate_text(prompt)
            self.text_output.setPlainText(response)
        
        threading.Thread(target=generate_thread, daemon=True).start()
    
    def on_voice_input(self):
        def voice_thread():
            text = self.ai.process_voice()
            if text:
                self.text_input.setPlainText(text)
        
        threading.Thread(target=voice_thread, daemon=True).start()
    
    def start_fine_tuning(self):
        def fine_tune_thread():
            # Example fine-tuning setup
            data = {"text": ["sample text 1", "sample text 2"]} 
            dataset = Dataset.from_dict(data)
            
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none"
            )
            
            model = get_peft_model(self.ai.model, peft_config)
            
            training_args = TrainingArguments(
                output_dir="./results",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                num_train_epochs=1,
                learning_rate=3e-4,
                fp16=True
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=self.ai.tokenizer
            )
            
            trainer.train()
            self.text_output.append("\nFine-tuning complete!")
        
        threading.Thread(target=fine_tune_thread, daemon=True).start()

# ======================
# MAIN EXECUTION 
# ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Verify GPU availability
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Performance will be severely limited.")
    
    window = AIWindow()
    window.show()
    sys.exit(app.exec_())
