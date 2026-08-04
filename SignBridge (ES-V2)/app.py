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

# NOTE: Do NOT create a global hands instance here.
# Each thread (camera_loop, sign_loop) creates its own
# local Hands instance to avoid MediaPipe thread-safety crashes.

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

class EchoSignV2:

    def __init__(self, root):

        self.root = root

        self.root.title("SignBridge AI – Bridging Communication Between Deaf and Hearing Individuals")

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

        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:

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
                    (480, 480),
                    Image.LANCZOS
                )

                photo = ImageTk.PhotoImage(image)

                self.camera_label.after(
                    0,
                    lambda p=photo: self.update_camera(p)
                )

                time.sleep(0.033)  # ~30 fps — prevents frame queue backlog

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

        self.letter_label.config(text="—")
        self.confidence_label.config(text="—")
        self.word_label.config(text="—")

    def sign_loop(self):

        current_word = ""
        last_letter = ""
        last_time = time.time()
        last_seen_hand = time.time()

        confidence_threshold = 0.5
        space_threshold = 1.5

        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:

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

                        # ── update live display ────────
                        self.root.after(
                            0,
                            lambda lt=prediction, cf=confidence:
                            self.update_live_detection(lt, cf)
                        )

                        if (
                            prediction != last_letter
                            and
                            time.time() - last_time > 0.7
                        ):

                            current_word += prediction

                            # ── update word display ────
                            self.root.after(
                                0,
                                lambda w=current_word:
                                self.word_label.config(text=w)
                            )

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

                    # ── clear word display after submit ─
                    self.root.after(
                        0,
                        lambda: self.word_label.config(text="—")
                    )

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

    def update_live_detection(self, letter, confidence):

        self.letter_label.config(text=letter)
        self.confidence_label.config(
            text=f"{int(confidence * 100)}%"
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
    # UI
    # ==================================================

    def create_ui(self):

        # ── Root grid: row 0 = header  (~5%)
        #               row 1 = main display (~85%)
        #               row 2 = bottom controls (~10%)
        # weights  1 : 17 : 2  → ~5% / ~85% / ~10%
        self.root.configure(bg="white")
        self.root.grid_rowconfigure(0, weight=1)    # header  –  ~5%
        self.root.grid_rowconfigure(1, weight=17)   # main    – ~85%
        self.root.grid_rowconfigure(2, weight=2)    # bottom  – ~10%
        self.root.grid_columnconfigure(0, weight=1)

        # ── Colour palette (white + blue) ─────────────
        ROOT_BG     = "white"
        HEADER_BG   = "#1565C0"        # deep blue
        HEADER_FG   = "white"
        SUBTITLE_FG = "#BBDEFB"        # light blue
        ACCENT      = "#1976D2"        # medium blue
        PANEL_BG    = "white"
        PANEL_BORDER= "#1976D2"
        LABEL_FG    = "#0D47A1"        # dark blue
        TEXT_BG     = "#F0F4FF"        # very light blue tint
        TEXT_FG     = "#0D1B2A"        # near-black
        STATUS_BG   = "#E3F2FD"        # pale blue
        STATUS_FG   = "#0D47A1"
        BTN_START   = "#1565C0"
        BTN_STOP    = "#1976D2"
        BTN_UTIL    = "#42A5F5"

        # ══════════════════════════════════════════════
        # ROW 0 – HEADER  (10%)
        # ══════════════════════════════════════════════

        header_frame = tk.Frame(
            self.root,
            bg=HEADER_BG
        )
        header_frame.grid(
            row=0, column=0,
            sticky="nsew",
            padx=0, pady=0
        )
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_rowconfigure(0, weight=1)
        header_frame.grid_rowconfigure(1, weight=1)
        header_frame.grid_rowconfigure(2, weight=0)

        tk.Label(
            header_frame,
            text="SignBridge AI",
            font=("Segoe UI", 16, "bold"),
            fg=HEADER_FG,
            bg=HEADER_BG
        ).grid(row=0, column=0, sticky="s", padx=10, pady=(3, 0))

        tk.Label(
            header_frame,
            text="Bridging Communication Between Deaf and Hearing Individuals",
            font=("Segoe UI", 8),
            fg=SUBTITLE_FG,
            bg=HEADER_BG
        ).grid(row=1, column=0, sticky="n", padx=10, pady=(0, 3))

        # thin bottom accent line
        tk.Frame(
            header_frame,
            bg="#42A5F5",
            height=3
        ).grid(row=2, column=0, sticky="ew")

        # ══════════════════════════════════════════════
        # ROW 1 – MAIN DISPLAY AREA  (80%)
        #   col 0 = Sign Input   20%  → weight 2
        #   col 1 = Camera       30%  → weight 3
        #   col 2 = Sign Display 30%  → weight 3
        #   col 3 = Speech Input 20%  → weight 2
        # ══════════════════════════════════════════════

        main_frame = tk.Frame(
            self.root,
            bg=ROOT_BG
        )
        main_frame.grid(
            row=1, column=0,
            sticky="nsew",
            padx=8, pady=4
        )

        main_frame.grid_columnconfigure(0, weight=2)
        main_frame.grid_columnconfigure(1, weight=3)
        main_frame.grid_columnconfigure(2, weight=3)
        main_frame.grid_columnconfigure(3, weight=2)
        main_frame.grid_rowconfigure(0, weight=1)

        def make_panel(parent, col, title):
            """Styled LabelFrame that never resizes to its children."""
            lf = tk.LabelFrame(
                parent,
                text=f"  {title}  ",
                font=("Segoe UI", 11, "bold"),
                fg=LABEL_FG,
                bg=PANEL_BG,
                bd=2,
                relief="groove",
                labelanchor="n"
            )
            lf.grid(
                row=0, column=col,
                sticky="nsew",
                padx=6, pady=6
            )
            lf.grid_propagate(False)
            lf.grid_rowconfigure(0, weight=1)
            lf.grid_columnconfigure(0, weight=1)
            return lf

        # ── col 0 : Sign Input ─────────────────────────
        left_frame = make_panel(main_frame, 0, "✋  Sign Input")

        tk.Label(
            left_frame,
            text="Sign → Text  |  Sign → Speech",
            font=("Segoe UI", 8),
            fg=ACCENT,
            bg=PANEL_BG
        ).grid(row=0, column=0, sticky="n", pady=(4, 0))

        # ── Live detection readout ─────────────────────
        live_frame = tk.Frame(left_frame, bg=PANEL_BG)
        live_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 0))
        live_frame.grid_columnconfigure(0, weight=1)
        live_frame.grid_columnconfigure(1, weight=1)

        tk.Label(
            live_frame,
            text="Letter",
            font=("Segoe UI", 8),
            fg="#888",
            bg=PANEL_BG
        ).grid(row=0, column=0)

        tk.Label(
            live_frame,
            text="Confidence",
            font=("Segoe UI", 8),
            fg="#888",
            bg=PANEL_BG
        ).grid(row=0, column=1)

        self.letter_label = tk.Label(
            live_frame,
            text="—",
            font=("Segoe UI", 32, "bold"),
            fg=LABEL_FG,
            bg=PANEL_BG,
            width=3
        )
        self.letter_label.grid(row=1, column=0, pady=(0, 4))

        self.confidence_label = tk.Label(
            live_frame,
            text="—",
            font=("Segoe UI", 14, "bold"),
            fg=ACCENT,
            bg=PANEL_BG
        )
        self.confidence_label.grid(row=1, column=1, pady=(0, 4))

        # ── word being built ───────────────────────────
        tk.Label(
            left_frame,
            text="Current Word",
            font=("Segoe UI", 8),
            fg="#888",
            bg=PANEL_BG
        ).grid(row=2, column=0, columnspan=2, sticky="n", pady=(4, 0))

        self.word_label = tk.Label(
            left_frame,
            text="—",
            font=("Segoe UI", 15, "bold"),
            fg="#1565C0",
            bg="#E3F2FD",
            relief="flat",
            anchor="center",
            padx=6,
            pady=4
        )
        self.word_label.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(2, 4))

        tk.Frame(left_frame, bg="#BBDEFB", height=1).grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=8
        )

        self.sign_text = tk.Text(
            left_frame,
            font=("Segoe UI", 13),
            bg=TEXT_BG,
            fg=TEXT_FG,
            insertbackground=TEXT_FG,
            relief="flat",
            wrap="word",
            bd=0
        )
        self.sign_text.grid(
            row=5, column=0,
            sticky="nsew",
            padx=8, pady=(4, 8)
        )
        left_frame.grid_rowconfigure(5, weight=1)

        sign_scroll = ttk.Scrollbar(
            left_frame,
            orient="vertical",
            command=self.sign_text.yview
        )
        sign_scroll.grid(row=5, column=1, sticky="ns", pady=(4, 8))
        self.sign_text.configure(yscrollcommand=sign_scroll.set)
        left_frame.grid_columnconfigure(1, weight=0)

        # ── col 1 : Camera Window ──────────────────────
        camera_outer = make_panel(main_frame, 1, "📷  Camera Window")

        camera_container = tk.Frame(
            camera_outer,
            bg="black",
            width=480,
            height=480
        )
        camera_container.grid(
            row=0, column=0,
            padx=8, pady=8
        )
        camera_container.grid_propagate(False)

        self.camera_label = tk.Label(
            camera_container,
            bg="black",
            text="No Feed",
            fg="#888899",
            font=("Segoe UI", 10)
        )
        self.camera_label.place(
            relx=0, rely=0,
            relwidth=1, relheight=1
        )

        # ── col 2 : Sign Display ───────────────────────
        sign_outer = make_panel(main_frame, 2, "🖼  Sign Display")

        sign_container = tk.Frame(
            sign_outer,
            bg=TEXT_BG,
            width=480,
            height=480
        )
        sign_container.grid(
            row=0, column=0,
            padx=8, pady=8
        )
        sign_container.grid_propagate(False)

        self.sign_image_label = tk.Label(
            sign_container,
            text="Waiting for\nTranslation...",
            font=("Segoe UI", 13),
            fg="#90A4AE",
            bg=TEXT_BG
        )
        self.sign_image_label.place(
            relx=0, rely=0,
            relwidth=1, relheight=1
        )

        # ── col 3 : Speech Input ───────────────────────
        right_frame = make_panel(main_frame, 3, "🎤  Speech Input")

        tk.Label(
            right_frame,
            text="Speech → Text  |  Speech → Sign",
            font=("Segoe UI", 8),
            fg=ACCENT,
            bg=PANEL_BG
        ).grid(row=0, column=0, sticky="n", pady=(4, 0))

        self.speech_text = tk.Text(
            right_frame,
            font=("Segoe UI", 13),
            bg=TEXT_BG,
            fg=TEXT_FG,
            insertbackground=TEXT_FG,
            relief="flat",
            wrap="word",
            bd=0
        )
        self.speech_text.grid(
            row=1, column=0,
            sticky="nsew",
            padx=8, pady=(4, 8)
        )
        right_frame.grid_rowconfigure(1, weight=1)

        speech_scroll = ttk.Scrollbar(
            right_frame,
            orient="vertical",
            command=self.speech_text.yview
        )
        speech_scroll.grid(row=1, column=1, sticky="ns", pady=(4, 8))
        self.speech_text.configure(yscrollcommand=speech_scroll.set)
        right_frame.grid_columnconfigure(1, weight=0)

        # ══════════════════════════════════════════════
        # ROW 2 – BOTTOM SECTION  (10%)
        # ══════════════════════════════════════════════

        bottom_frame = tk.Frame(
            self.root,
            bg=STATUS_BG
        )
        bottom_frame.grid(
            row=2, column=0,
            sticky="nsew",
            padx=0, pady=0
        )
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_rowconfigure(0, weight=0)   # top accent line
        bottom_frame.grid_rowconfigure(1, weight=1)   # status indicators
        bottom_frame.grid_rowconfigure(2, weight=1)   # buttons

        # top accent line
        tk.Frame(
            bottom_frame,
            bg=ACCENT,
            height=3
        ).grid(row=0, column=0, sticky="ew")

        # ── Status row ────────────────────────────────
        status_outer = tk.Frame(
            bottom_frame,
            bg=STATUS_BG
        )
        status_outer.grid(row=1, column=0, sticky="nsew", padx=8)
        status_outer.grid_columnconfigure(0, weight=1)
        status_outer.grid_columnconfigure(1, weight=1)
        status_outer.grid_columnconfigure(2, weight=1)

        self.camera_status = tk.Label(
            status_outer,
            text="🔴  Camera Disconnected",
            font=("Segoe UI", 9, "bold"),
            fg=STATUS_FG,
            bg=STATUS_BG,
            anchor="center"
        )
        self.camera_status.grid(row=0, column=0, sticky="ew", pady=2)

        self.sign_status = tk.Label(
            status_outer,
            text="🔴  Sign Detection Stopped",
            font=("Segoe UI", 9, "bold"),
            fg=STATUS_FG,
            bg=STATUS_BG,
            anchor="center"
        )
        self.sign_status.grid(row=0, column=1, sticky="ew", pady=2)

        self.speech_status = tk.Label(
            status_outer,
            text="🔴  Speech Detection Stopped",
            font=("Segoe UI", 9, "bold"),
            fg=STATUS_FG,
            bg=STATUS_BG,
            anchor="center"
        )
        self.speech_status.grid(row=0, column=2, sticky="ew", pady=2)

        # ── Button row ────────────────────────────────
        button_frame = tk.Frame(
            bottom_frame,
            bg=STATUS_BG
        )
        button_frame.grid(row=2, column=0, pady=(0, 4))

        BTN_STYLE = {
            "font"           : ("Segoe UI", 9, "bold"),
            "relief"         : "flat",
            "cursor"         : "hand2",
            "padx"           : 12,
            "pady"           : 5,
            "bd"             : 0
        }

        def styled_btn(parent, text, cmd, col, color, hover):
            b = tk.Button(
                parent,
                text=text,
                command=cmd,
                bg=color,
                fg="white",
                activebackground=hover,
                activeforeground="white",
                **BTN_STYLE
            )
            b.grid(row=0, column=col, padx=5)
            return b

        styled_btn(button_frame, "▶ Start Camera",  self.start_camera,  0, "#1565C0", "#1976D2")
        styled_btn(button_frame, "■ Stop Camera",   self.stop_camera,   1, "#1976D2", "#42A5F5")
        styled_btn(button_frame, "▶ Start Sign",    self.start_sign,    2, "#1565C0", "#1976D2")
        styled_btn(button_frame, "■ Stop Sign",     self.stop_sign,     3, "#1976D2", "#42A5F5")
        styled_btn(button_frame, "▶ Start Speech",  self.start_speech,  4, "#1565C0", "#1976D2")
        styled_btn(button_frame, "■ Stop Speech",   self.stop_speech,   5, "#1976D2", "#42A5F5")
        styled_btn(button_frame, "🗑 Clear Text",    self.clear_text,    6, "#42A5F5", "#64B5F6")
        styled_btn(button_frame, "📂 Clear History", self.clear_history, 7, "#42A5F5", "#64B5F6")


root = tk.Tk()

app = EchoSignV2(root)

root.protocol(
    "WM_DELETE_WINDOW",
    app.on_close
)

root.mainloop()