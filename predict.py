import cv2
import mediapipe as mp
import numpy as np
import joblib
import math
import time
import pyttsx3
import difflib
import datetime
import pickle


def save_conversation(text):
    with open("conversation.txt", "a") as f:
        f.write(text + "\n")

def save_conversation(text):
    with open("conversation.txt", "a") as f:
        time_now = datetime.datetime.now().strftime("%H:%M:%S")
        f.write(f"[{time_now}] {text}\n")

# LOAD MODEL 
model = joblib.load("model.pkl")

#TTS
engine = pyttsx3.init()

#DICTIONARY 
dictionary = [
    "HELLO", "HELP", "YES", "NO", "STOP", "GO", "COME",
    "PLEASE", "THANKS", "HI", "YOU", "ME"
]

def correct_word(word):
    matches = difflib.get_close_matches(word, dictionary, n=1, cutoff=0.6)
    return matches[0] if matches else word

#MEDIAPIPE 
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

# CAMERA --------
cap = cv2.VideoCapture(0)

#TEXT SYSTEM 
current_text = ""
current_word = ""
last_letter = ""
last_time = time.time()
cooldown = 1.0

#PAUSE DETECTION 
last_seen_hand = time.time()
space_threshold = 2.0
sentence_threshold = 4.0

while True:
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

            #Confidence filter
            probs = model.predict_proba([normalized])[0]
            confidence = max(probs)

            if confidence > 0.8:
                prediction = model.predict([normalized])[0]

                #Letter add 
                if prediction == last_letter:
                    if time.time() - last_time > cooldown:
                        current_word += prediction
                        last_time = time.time()
                else:
                    last_letter = prediction
                    last_time = time.time()

            cv2.putText(img, f"{prediction} ({confidence:.2f})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    #Auto-space
    if not hand_detected:
        pause_time = time.time() - last_seen_hand

        #word
        if pause_time > space_threshold and current_word != "":
            corrected = correct_word(current_word)
            current_text += corrected + " "
            print("Text:", current_text)
            current_word = ""
            time.sleep(0.5)

        # Sentence  → auto-speak
        if pause_time > sentence_threshold and current_text.strip() != "":
            print("🔊 Speaking:", current_text)
            print("🔊 Speaking:", current_text)

            engine.say(current_text)
            engine.runAndWait()

            # Save Conversation
            save_conversation(current_text)

            current_text = ""
            time.sleep(1)

    # --Display
    cv2.putText(img, f"Word: {current_word}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.putText(img, f"Text: {current_text}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.putText(img, "C = Clear | ESC = Exit", (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow("Prediction", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        current_text = ""
        current_word = ""
        print("🧹 Cleared")

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()