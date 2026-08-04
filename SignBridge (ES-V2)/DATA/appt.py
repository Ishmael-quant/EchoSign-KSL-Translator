import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

import cv2
import threading
import time
import os
import json
import math
import difflib


import joblib
import mediapipe as mp 
import sounddevice as sd

import edge_tts
import asyncio
import pygame

from vosk import Model, KaldiRecognizer
from datetime import datetime

# ==================================================
# CONFIG
# ==================================================

MODEL_PATH = "vosk-model-small-en-us-0.15"
SIGN_FOLDER = "Signs"

# ==================================================
# LOAD AI MODEL
# ==================================================

model = joblib.load("model.pkl")

# ==================================================
# MEDIAPIPE
# ==================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==================================================
# TTS
# ==================================================
pygame.mixer.init()

# ==================================================
# DICTIONARY
# ==================================================

dictionary = [
    "hello",
    "help",
    "yes",
    "no",
    "stop",
    "go",
    "come",
    "please",
    "thanks",
    "hi",
    "you",
    "me"
]

def correct_text(text):

    words = text.split()

    corrected = []

    for word in words:

        match = difflib.get_close_matches(
            word,
            dictionary,
            n=1,
            cutoff=0.6
        )

        corrected.append(
            match[0] if match else word
        )

    return " ".join(corrected)

# ==================================================
# SAVE CONVERSATION
# ==================================================

def save_conversation(text):

    with open(
        "conversation.txt",
        "a",
        encoding="utf-8"
    ) as f:

        timestamp = datetime.now().strftime(
            "%H:%M:%S"
        )

        f.write(
            f"[{timestamp}] {text}\n"
        )

# ==================================================
# EDGE TTS
# ==================================================

async def generate_audio(text, filename):

    communicate = edge_tts.Communicate(
        text,
        voice="en-US-AriaNeural"
    )

    await communicate.save(filename)

# ==================================================
# APP
# ==================================================

class SignBridgeV2:

    def __init__(self, root):

        self.root = root

        self.root.title("SignBridge AI V2")

        try:
            self.root.state("zoomed")
        except:
            self.root.geometry("1400x800")

        # ---------------- STATES ----------------

        self.running_camera = False
        self.running_sign = False
        self.running_speech = False

        self.cap = None

        self.current_sign_image = None

        self.create_ui()

    # ==================================================
    # CAMERA
    # ==================================================

    def start_camera(self):

        if self.running_camera:
            return

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Could not open camera."
            )
            return

        self.running_camera = True

        self.camera_status.config(
            text="🟢 Camera Connected"
        )

        threading.Thread(
            target=self.camera_loop,
            daemon=True
        ).start()

    def stop_camera(self):

        self.running_camera = False

        if self.cap:
            self.cap.release()
            self.cap = None

        self.camera_status.config(
            text="🔴 Camera Disconnected"
        )

    def camera_loop(self):

        while self.running_camera:

            success, frame = self.cap.read()

            if not success:
                continue

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb)

            if results.multi_hand_landmarks:

                for handLms in results.multi_hand_landmarks:

                    mp_draw.draw_landmarks(
                        frame,
                        handLms,
                        mp_hands.HAND_CONNECTIONS
                    )
                    lmList = [
                       [lm.x, lm.y, lm.z]
                       for lm in handLms.landmark
                    ]

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(rgb)

            image = image.resize(
                (900, 550)
            )

            photo = ImageTk.PhotoImage(image)

            self.camera_label.after(
                0,
                lambda p=photo: self.update_camera(p)
            )

            time.sleep(0.01)

    def update_camera(self, photo):

        self.camera_label.config(
            image=photo
        )

        self.camera_label.image = photo

    # ==================================================
    # SPEECH
    # ==================================================

    def speak_text(self, text):

        try:

            filename = (
                f"Audio/{int(time.time()*1000)}.mp3"
            )

            asyncio.run(
                generate_audio(
                    text,
                    filename
                )
            )

            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            if os.path.exists(filename):
                os.remove(filename)

        except Exception as e:

            print(
                "Edge TTS Error:",
                e
            )

    # ==================================================
    # SIGN DETECTION
    # ==================================================

    def start_sign(self):

        if self.running_sign:
            return

        self.running_sign = True

        self.sign_status.config(
            text="🟢 Sign Detection Active"
        )

        threading.Thread(
            target=self.sign_loop,
            daemon=True
        ).start()

    def stop_sign(self):

        self.running_sign = False

        self.sign_status.config(
            text="🔴 Sign Detection Stopped"
        )

    def sign_loop(self):

        current_word = ""
        last_letter = ""
        last_time = time.time()
        last_seen_hand = time.time()

        confidence_threshold = 0.5
        space_threshold = 1.5

        while self.running_sign:

            if not self.cap:
                time.sleep(0.1)
                continue

            success, frame = self.cap.read()

            if not success:
                continue

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb)

            hand_detected = False

            if results.multi_hand_landmarks:

                hand_detected = True
                last_seen_hand = time.time()

                for handLms in results.multi_hand_landmarks:

                    lmList = [
                        [lm.x, lm.y, lm.z]
                        for lm in handLms.landmark
                    ]

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

                    probs = model.predict_proba(
                        [normalized]
                    )[0]

                    confidence = max(probs)

                    if confidence > confidence_threshold:

                        prediction = model.predict(
                            [normalized]
                        )[0]

                        if (
                            prediction != last_letter
                            and
                            time.time() - last_time > 0.7
                        ):

                            current_word += prediction

                            last_letter = prediction

                            last_time = time.time()

            if not hand_detected:

                pause_time = (
                    time.time() - last_seen_hand
                )

                if (
                    pause_time > space_threshold
                    and
                    current_word != ""
                ):

                    self.root.after(
                        0,
                        lambda w=current_word:
                        self.add_sign_text(w)
                    )

                    threading.Thread(
                        target=self.speak_text,
                        args=(current_word,),
                        daemon=True
                    ).start()

                    save_conversation(
                        current_word
                    )

                    current_word = ""

                    time.sleep(0.3)

            time.sleep(0.01)

    def add_sign_text(self, text):

        self.sign_text.insert(
            tk.END,
            text + " "
        )

        self.sign_text.see(
            tk.END
        )

    # ==================================================
    # SPEECH RECOGNITION
    # ==================================================

    def start_speech(self):

        if self.running_speech:
            return

        self.running_speech = True

        self.speech_status.config(
            text="🟢 Speech Detection Active"
        )

        threading.Thread(
            target=self.speech_loop,
            daemon=True
        ).start()

    def stop_speech(self):

        self.running_speech = False

        self.speech_status.config(
            text="🔴 Speech Detection Stopped"
        )

    def speech_loop(self):

        model_vosk = Model(
            MODEL_PATH
        )

        recognizer = KaldiRecognizer(
            model_vosk,
            16000
        )

        sentence = ""

        last_speech_time = time.time()

        def callback(
            indata,
            frames,
            time_info,
            status
        ):

            nonlocal sentence
            nonlocal last_speech_time

            if recognizer.AcceptWaveform(
                bytes(indata)
            ):

                result = json.loads(
                    recognizer.Result()
                )

                text = result.get(
                    "text",
                    ""
                )

                if text:

                    sentence += (
                        " " + text
                    )

                    last_speech_time = (
                        time.time()
                    )

        with sd.RawInputStream(

            samplerate=16000,

            blocksize=8000,

            dtype="int16",

            channels=1,

            callback=callback

        ):

            while self.running_speech:

                time.sleep(0.1)

                if (
                    sentence != ""
                    and
                    time.time() -
                    last_speech_time > 3
                ):

                    final_text = (
                        correct_text(
                            sentence
                            .lower()
                            .strip()
                        )
                    )

                    self.root.after(
                        0,
                        lambda t=final_text:
                        self.add_speech_text(t)
                    )

                    save_conversation(
                        final_text
                    )

                    threading.Thread(
                        target=self.speak_text,
                        args=(final_text,),
                        daemon=True
                    ).start()

                    threading.Thread(
                        target=self.show_signs,
                        args=(final_text.upper(),),
                        daemon=True
                    ).start()

                    sentence = ""

    def add_speech_text(self, text):

        self.speech_text.insert(
            tk.END,
            text + "\n"
        )

        self.speech_text.see(
            tk.END
        )

    # ==================================================
    # SIGN DISPLAY
    # ==================================================

    def show_signs(self, text):

        for letter in text:

            if letter == " ":

                time.sleep(0.7)

                continue

            path = os.path.join(
                SIGN_FOLDER,
                f"{letter}.png"
            )

            if not os.path.exists(
                path
            ):
                continue

            image = Image.open(
                path
            )

            image = image.resize(
                (250, 250)
            )

            photo = ImageTk.PhotoImage(
                image
            )

            self.root.after(
                0,
                lambda p=photo:
                self.update_sign_image(p)
            )

            time.sleep(0.5)

    def update_sign_image(self, photo):

        self.sign_image_label.config(
            image=photo,
            text=""
        )

        self.sign_image_label.image = photo

    # ==================================================
    # UTILITIES
    # ==================================================

    def clear_text(self):

        self.sign_text.delete(
            "1.0",
            tk.END
        )

        self.speech_text.delete(
            "1.0",
            tk.END
        )

    def clear_history(self):

        open(
            "conversation.txt",
            "w",
            encoding="utf-8"
        ).close()

        messagebox.showinfo(
            "Success",
            "Conversation history cleared."
        )

    def on_close(self):

        self.running_camera = False
        self.running_sign = False
        self.running_speech = False

        if self.cap:
            self.cap.release()

        self.root.destroy()

    # ==================================================
    # UI - ONLY LAYOUT FIXED, NOTHING ELSE CHANGED
    # ==================================================

    def create_ui(self):

        # ---------------- TITLE ----------------

        title = tk.Label(
            self.root,
            text="SignBridge AI V2",
            font=("Arial", 22, "bold")
        )

        title.pack(pady=10)

        # ---------------- MAIN FRAME ----------------

        main_frame = ttk.Frame(self.root)

        main_frame.pack(
            fill="both",
            expand=True
        )

        # ==================================================
        # LEFT PANEL (15%)
        # ==================================================

        left_frame = ttk.LabelFrame(
            main_frame,
            text="Sign Output"
        )

        left_frame.pack(
            side="left",
            fill="both",
            expand=False,
            padx=5,
            pady=10
        )
        
        left_frame.pack_propagate(False)
        left_frame.config(width=200)

        self.sign_text = tk.Text(
            left_frame,
            font=("Arial", 14)
        )

        self.sign_text.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5
        )

        # ==================================================
        # CENTER PANEL (70%)
        # ==================================================

        center_frame = ttk.Frame(
            main_frame
        )

        center_frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=5,
            pady=10
        )

        # ---------------- CAMERA ----------------

        camera_frame = ttk.LabelFrame(
            center_frame,
            text="Camera Feed"
        )

        camera_frame.pack(
            fill="both",
            expand=True,
            pady=(0, 5)
        )

        self.camera_label = tk.Label(
            camera_frame,
            bg="black"
        )

        self.camera_label.pack(
            fill="both",
            expand=True
        )

        # ---------------- SIGN DISPLAY ----------------

        sign_display_frame = ttk.LabelFrame(
            center_frame,
            text="Sign Display"
        )

        sign_display_frame.pack(
            fill="x",
            pady=(5, 0)
        )

        self.sign_image_label = tk.Label(
            sign_display_frame,
            text="Waiting for Translation...",
            font=("Arial", 14),
            height=4
        )

        self.sign_image_label.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        # ==================================================
        # RIGHT PANEL (15%)
        # ==================================================

        right_frame = ttk.LabelFrame(
            main_frame,
            text="Speech Output"
        )

        right_frame.pack(
            side="left",
            fill="both",
            expand=False,
            padx=5,
            pady=10
        )
        
        right_frame.pack_propagate(False)
        right_frame.config(width=200)

        self.speech_text = tk.Text(
            right_frame,
            font=("Arial", 14)
        )

        self.speech_text.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5
        )

        # ==================================================
        # STATUS BAR
        # ==================================================

        status_frame = ttk.LabelFrame(
            self.root,
            text="System Status"
        )

        status_frame.pack(
            fill="x",
            padx=10,
            pady=5
        )

        self.camera_status = tk.Label(
            status_frame,
            text="🔴 Camera Disconnected",
            font=("Arial", 11)
        )

        self.camera_status.pack(
            side="left",
            padx=20
        )

        self.sign_status = tk.Label(
            status_frame,
            text="🔴 Sign Detection Stopped",
            font=("Arial", 11)
        )

        self.sign_status.pack(
            side="left",
            padx=20
        )

        self.speech_status = tk.Label(
            status_frame,
            text="🔴 Speech Detection Stopped",
            font=("Arial", 11)
        )

        self.speech_status.pack(
            side="left",
            padx=20
        )

        # ==================================================
        # BUTTONS
        # ==================================================

        button_frame = ttk.Frame(
            self.root
        )

        button_frame.pack(
            pady=10
        )

        ttk.Button(
            button_frame,
            text="Start Camera",
            command=self.start_camera
        ).grid(row=0, column=0, padx=10)

        ttk.Button(
            button_frame,
            text="Stop Camera",
            command=self.stop_camera
        ).grid(row=0, column=1, padx=10)

        ttk.Button(
            button_frame,
            text="Start Sign",
            command=self.start_sign
        ).grid(row=0, column=2, padx=10)

        ttk.Button(
            button_frame,
            text="Stop Sign",
            command=self.stop_sign
        ).grid(row=0, column=3, padx=10)

        ttk.Button(
            button_frame,
            text="Start Speech",
            command=self.start_speech
        ).grid(row=0, column=4, padx=10)

        ttk.Button(
            button_frame,
            text="Stop Speech",
            command=self.stop_speech
        ).grid(row=0, column=5, padx=10)

        ttk.Button(
            button_frame,
            text="Clear Text",
            command=self.clear_text
        ).grid(row=0, column=6, padx=10)

        ttk.Button(
            button_frame,
            text="Clear History",
            command=self.clear_history
        ).grid(row=0, column=7, padx=10)


root = tk.Tk()

app = SignBridgeV2(root)

root.protocol(
    "WM_DELETE_WINDOW",
    app.on_close
)

root.mainloop()