import cv2
import mediapipe as mp
import csv
import math
import os

# -------- SETTINGS --------
DATA_FILE = "dataset.csv"

# Create file if it doesn't exist
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as f:
        pass

# Camera
cap = cv2.VideoCapture(0)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

# Ask for label
label = input("Enter label (e.g., hello): ")

saving = False  # control saving mode

with open(DATA_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        success, img = cap.read()
        if not success:
            print("❌ Camera not working")
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        # Draw landmarks if detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:

                # DRAW LANDMARKS
                mp_draw.draw_landmarks(
                    img, handLms, mp_hands.HAND_CONNECTIONS
                )

                # Extract landmarks
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

                # Save data continuously
                if saving:
                    writer.writerow(normalized + [label])
                    print("✅ Saving:", label)

        # Show instructions on screen
        cv2.putText(img, "Press S to START saving", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(img, "Press E to STOP saving", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(img, "Press ESC to EXIT", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow("Data Collection", img)

        key = cv2.waitKey(1)

        if key != -1:
            print("Key pressed:", key)  # DEBUG

        if key == ord('s'):
            saving = True
            print("🟢 Started Saving")

        elif key == ord('e'):
            saving = False
            print("🛑 Stopped Saving")

cap.release()
cv2.destroyAllWindows()