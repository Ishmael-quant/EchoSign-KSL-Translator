import tkinter as tk
from tkinter import ttk
import threading
import cv2
import time
import os
import joblib
import mediapipe as mp
import pyttsx3
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from datetime import datetime
import math
import difflib

# -------- LOAD MODEL --------
model = joblib.load("model.pkl")

dictionary = [
    "hello", "help", "yes", "no", "stop", "go", "come",
    "please", "thanks", "hi", "you", "me"
]
def correct_text(text):
    words = text.split()
    corrected = []

    for word in words:
        match = difflib.get_close_matches(word, dictionary, n=1, cutoff=0.6)
        corrected.append(match[0] if match else word)

    return " ".join(corrected)

# -------- MEDIAPIPE --------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# -------- TTS --------
engine = pyttsx3.init()

# -------- PATHS --------
SIGN_FOLDER = "Signs"
MODEL_PATH = "vosk-model-small-en-us-0.15"

# -------- SAVE CONVERSATION --------
def save_conversation(text):
    with open("conversation.txt", "a") as f:
        time_now = datetime.now().strftime("%H:%M:%S")
        f.write(f"[{time_now}] {text}\n")

# -------- APP --------
class App:

    def __init__(self, root):
        self.root = root
        self.root.title("EchoSign AI")
        self.root.geometry("800x500")

        self.running_sign = False
        self.running_speech = False

        self.create_ui()

    # -------- UI --------
    def create_ui(self):

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # Sign Output
        sign_frame = ttk.LabelFrame(main_frame, text="Sign Output")
        sign_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.sign_text = tk.Text(sign_frame)
        self.sign_text.pack(fill="both", expand=True)

        # Speech Output
        speech_frame = ttk.LabelFrame(main_frame, text="Speech Output")
        speech_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.speech_text = tk.Text(speech_frame)
        self.speech_text.pack(fill="both", expand=True)

        # Controls
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)

        ttk.Button(control_frame, text="Start Sign", command=self.start_sign).grid(row=0, column=0, padx=10)
        ttk.Button(control_frame, text="Stop Sign", command=self.stop_sign).grid(row=0, column=1, padx=10)

        ttk.Button(control_frame, text="Start Speech", command=self.start_speech).grid(row=1, column=0, padx=10)
        ttk.Button(control_frame, text="Stop Speech", command=self.stop_speech).grid(row=1, column=1, padx=10)

    # -------- SIGN LOOP --------
    def sign_loop(self):
        cap = cv2.VideoCapture(0)

        current_word = ""
        last_letter = ""
        last_time = time.time()
        last_seen_hand = time.time()

        cooldown = 1.0
        space_threshold = 2.0

        confidence_threshold = 0.6

        while self.running_sign:
            success, img = cap.read()
            if not success:
                break

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            prediction = ""
            hand_detected = False

            if results.multi_hand_landmarks:
                hand_detected = True
                last_seen_hand = time.time()

                for handLms in results.multi_hand_landmarks:

                    mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

                    lmList = []
                    for lm in handLms.landmark:
                        lmList.append([lm.x, lm.y, lm.z])

                    # Normalize
                    wrist = lmList[0]
                    ref = lmList[12]

                    scale = math.sqrt(
                        (ref[0] - wrist[0])**2 +
                        (ref[1] - wrist[1])**2 +
                        (ref[2] - wrist[2])**2
                    )

                    normalized = []
                    for lm in lmList:
                        nx = (lm[0] - wrist[0]) / scale
                        ny = (lm[1] - wrist[1]) / scale
                        nz = (lm[2] - wrist[2]) / scale
                        normalized.extend([nx, ny, nz])

                    probs = model.predict_proba([normalized])[0]
                    confidence = max(probs)

                    if confidence > confidence_threshold:
                        prediction = model.predict([normalized])[0]

                        if prediction != last_letter and time.time() - last_time > 0.8:
                            current_word += prediction
                            last_letter = prediction
                            last_time = time.time()

                    cv2.putText(img, f"{prediction} ({confidence:.2f})",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if not hand_detected:
                pause_time = time.time() - last_seen_hand

                if pause_time > space_threshold and current_word != "":
                    self.sign_text.insert(tk.END, current_word + " ")
                    self.sign_text.see(tk.END)

                    save_conversation(current_word)

                    current_word = ""
                    time.sleep(0.5)

            cv2.putText(img, f"Word: {current_word}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("Sign Detection", img)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def start_sign(self):
        if not self.running_sign:
            self.running_sign = True
            threading.Thread(target=self.sign_loop, daemon=True).start()

    def stop_sign(self):
        self.running_sign = False

    # -------- SPEECH LOOP --------
    def speech_loop(self):
        model_vosk = Model(MODEL_PATH)
        recognizer = KaldiRecognizer(model_vosk, 16000)

        sentence = ""
        last_speech_time = time.time()

        def callback(indata, frames, time_info, status):
            nonlocal sentence, last_speech_time

            if recognizer.AcceptWaveform(bytes(indata)):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")

                if text:
                    sentence += " " + text
                    last_speech_time = time.time()

        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype='int16',
            channels=1,
            callback=callback
        ):
            while self.running_speech:
                time.sleep(0.1)

                if sentence != "" and time.time() - last_speech_time > 3:
                    final_text = correct_text(sentence.strip())

                    self.speech_text.insert(tk.END, final_text + "\n")
                    self.speech_text.see(tk.END)

                    save_conversation(final_text)

                    self.show_signs(final_text.upper())

                    sentence = ""

    def start_speech(self):
        if not self.running_speech:
            self.running_speech = True
            threading.Thread(target=self.speech_loop, daemon=True).start()

    def stop_speech(self):
        self.running_speech = False

    # -------- SHOW SIGNS --------
    def show_signs(self, text):
        for letter in text:
            if letter == " ":
                time.sleep(1)
                continue

            path = os.path.join(SIGN_FOLDER, f"{letter}.png")
            if not os.path.exists(path):
                continue

            img = cv2.imread(path)
            img = cv2.resize(img, (400, 400))

            cv2.imshow("Sign Display", img)
            cv2.waitKey(600)

        cv2.destroyAllWindows()


# -------- RUN --------
root = tk.Tk()
app = App(root)
root.mainloop()