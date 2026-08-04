import tkinter as tk
from tkinter import ttk
import threading
import cv2
import time
import os
import joblib
import mediapipe as mp
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from datetime import datetime
import math
import difflib
import queue
import asyncio
import edge_tts

# -------- SPEAK SAFE (Modified for Edge TTS) --------
speech_queue = queue.Queue()
loop = asyncio.new_event_loop()

async def speak_async(text):
    """Async function to speak using edge-tts"""
    try:
        communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
        await communicate.save("temp_speech.mp3")
        
        # Play the audio file
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load("temp_speech.mp3")
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)
            
        pygame.mixer.quit()
        
        # Clean up temp file
        if os.path.exists("temp_speech.mp3"):
            os.remove("temp_speech.mp3")
            
    except Exception as e:
        print(f"Edge TTS Error: {e}")

def speech_worker():
    """Worker thread for processing speech queue"""
    asyncio.set_event_loop(loop)
    
    while True:
        text = speech_queue.get()
        
        if text is None:  # Exit signal
            break
            
        print("Worker received:", text)
        
        try:
            print("Speaking:", text)
            # Run the async speak function
            loop.run_until_complete(speak_async(text))
            print("Finished:", text)
            
        except Exception as e:
            print("Speech Error:", e)
        
        speech_queue.task_done()

# Start speech worker thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# -------- LOAD MODEL --------
model = joblib.load("model.pkl")

# -------- DICTIONARY --------
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
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# -------- PATHS --------
SIGN_FOLDER = "Signs"
MODEL_PATH = "vosk-model-small-en-us-0.15"

# -------- SAVE --------
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

        sign_frame = ttk.LabelFrame(main_frame, text="Sign Output")
        sign_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.sign_text = tk.Text(sign_frame)
        self.sign_text.pack(fill="both", expand=True)

        speech_frame = ttk.LabelFrame(main_frame, text="Speech Output")
        speech_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.speech_text = tk.Text(speech_frame)
        self.speech_text.pack(fill="both", expand=True)

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

        confidence_threshold = 0.5
        space_threshold = 1.5   # better for real use

        while self.running_sign:
            success, img = cap.read()
            if not success:
               print("Camera frame lost")
               continue

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            prediction = ""
            hand_detected = False

            if results.multi_hand_landmarks:
                hand_detected = True
                last_seen_hand = time.time()

                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

                    lmList = [[lm.x, lm.y, lm.z] for lm in handLms.landmark]

                    wrist = lmList[0]
                    ref = lmList[12]

                    scale = math.sqrt(
                        (ref[0] - wrist[0])**2 +
                        (ref[1] - wrist[1])**2 +
                        (ref[2] - wrist[2])**2
                    )

                    normalized = []
                    for lm in lmList:
                        normalized.extend([
                            (lm[0] - wrist[0]) / scale,
                            (lm[1] - wrist[1]) / scale,
                            (lm[2] - wrist[2]) / scale
                        ])

                    probs = model.predict_proba([normalized])[0]
                    confidence = max(probs)

                    if confidence > confidence_threshold:
                        prediction = model.predict([normalized])[0]

                        if prediction != last_letter and time.time() - last_time > 0.7:
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

                    print("Speaking:", current_word)
                    speech_queue.put(current_word)

                    current_word = ""
                    time.sleep(0.3)

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

    # -------- FIXED SPEECH LOOP --------
    def speech_loop(self):
        model_vosk = Model(MODEL_PATH)
        recognizer = KaldiRecognizer(model_vosk, 16000)
        
        # Use a queue for audio data
        audio_queue = queue.Queue()
        
        def callback(indata, frames, time_info, status):
            audio_queue.put(bytes(indata))
        
        sentence = ""
        last_speech_time = time.time()
        
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype='int16',
            channels=1,
            callback=callback
        ):
            while self.running_speech:
                try:
                    # Get audio data with timeout
                    data = audio_queue.get(timeout=0.1)
                    
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "")
                        
                        if text:
                            print("RAW:", text)
                            
                            # Accumulate sentence
                            if sentence == "":
                                sentence = text
                            else:
                                # Check if this is continuation (within 2 seconds)
                                if time.time() - last_speech_time < 2.0:
                                    sentence += " " + text
                                else:
                                    # Process previous sentence
                                    if sentence.strip():
                                        final_text = correct_text(sentence.lower().strip())
                                        self.speech_text.insert(tk.END, final_text + "\n")
                                        self.speech_text.see(tk.END)
                                        save_conversation(final_text)
                                        print("Speaking:", final_text)
                                        speech_queue.put(final_text)
                                        self.show_signs(final_text.upper())
                                    
                                    # Start new sentence
                                    sentence = text
                            
                            last_speech_time = time.time()
                    else:
                        # Check for silence to process the sentence
                        if sentence and (time.time() - last_speech_time) > 2.0:
                            final_text = correct_text(sentence.lower().strip())
                            self.speech_text.insert(tk.END, final_text + "\n")
                            self.speech_text.see(tk.END)
                            save_conversation(final_text)
                            print("Speaking:", final_text)
                            speech_queue.put(final_text)
                            self.show_signs(final_text.upper())
                            sentence = ""
                            
                except queue.Empty:
                    # Check for silence when no audio is coming
                    if sentence and (time.time() - last_speech_time) > 2.0:
                        final_text = correct_text(sentence.lower().strip())
                        self.speech_text.insert(tk.END, final_text + "\n")
                        self.speech_text.see(tk.END)
                        save_conversation(final_text)
                        print("Speaking:", final_text)
                        speech_queue.put(final_text)
                        self.show_signs(final_text.upper())
                        sentence = ""
                    continue

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
                time.sleep(0.7)
                continue

            path = os.path.join(SIGN_FOLDER, f"{letter}.png")
            if not os.path.exists(path):
                continue

            img = cv2.imread(path)
            img = cv2.resize(img, (400, 400))

            cv2.imshow("Sign Display", img)
            cv2.waitKey(500)

        cv2.destroyAllWindows()

# -------- RUN --------
root = tk.Tk()
app = App(root)
root.mainloop()