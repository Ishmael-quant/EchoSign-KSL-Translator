import speech_recognition as sr
import cv2
import time
import os

#CHECK SIGNS FOLDER 
SIGN_FOLDER = "signs"

if not os.path.exists(SIGN_FOLDER):
    print("❌ 'signs' folder not found!")
    exit()

#SPEECH RECOGNIZER
recognizer = sr.Recognizer()

def listen_and_convert():
    with sr.Microphone() as source:
        print("🎤 Speak something...")
        recognizer.adjust_for_ambient_noise(source)

        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print("📝 You said:", text)
            return text.upper()
        except sr.WaitTimeoutError:
            print("⏱️ No speech detected")
        except sr.UnknownValueError:
            print("❌ Could not understand")
        except sr.RequestError:
            print("⚠️ API error")
    
    return ""

#SHOW SIGNS 
def show_signs(text):
    for letter in text:

        if letter == " ":
            print("⏸ Space detected")
            time.sleep(1)
            continue

        img_path = os.path.join(SIGN_FOLDER, f"{letter}.png")

        if not os.path.exists(img_path):
            print(f"⚠️ No sign image for: {letter}")
            continue

        img = cv2.imread(img_path)
        img = cv2.resize(img, (400, 400))

        if img is None:
            print(f"❌ Failed to load: {letter}")
            continue

        cv2.imshow("Sign Display", img)
        cv2.waitKey(800)  # show each letter for 0.8 sec

    cv2.destroyAllWindows()

#MAIN LOOP 
while True:
    text = listen_and_convert()

    if text:
        show_signs(text)

    print("\nPress ESC in window to stop or Ctrl+C in terminal\n")