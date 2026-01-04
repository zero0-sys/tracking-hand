import cv2, mediapipe as mp, numpy as np, os, time
from tensorflow.keras.models import load_model
import pyttsx3

# =====================
# LOAD MODEL & LABEL
# =====================
DATA_DIR = "data"
labels = sorted(os.listdir(DATA_DIR))   # ⬅️ HARUS SAMA DENGAN TRAINING
print("Labels:", labels)

model = load_model("gesture_lstm.h5")

# =====================
# TEXT TO SPEECH
# =====================
tts = pyttsx3.init()
tts.setProperty("rate", 150)
tts.setProperty("volume", 1.0)

def speak(text):
    tts.say(text)
    tts.runAndWait()

# =====================
# MEDIAPIPE
# =====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,                   # ⬅️ LEBIH STABIL
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
draw = mp.solutions.drawing_utils

# =====================
# CAMERA
# =====================
IP_CAM = "http://192.168.1.4:8080/video"
cap = cv2.VideoCapture(IP_CAM, cv2.CAP_FFMPEG)

# =====================
# STATE
# =====================
sequence = []

current_word = None
stable_count = 0

display_word = ""
sentence = []

last_hand_time = time.time()
last_spoken_word = ""

# =====================
# THRESHOLD & FILTER
# =====================
CONFIDENCE_THRESHOLD = 0.75
MIN_MARGIN = 0.20

STABLE_FRAMES = 8
SPECIAL_STABLE = 12

AUTO_CLEAR_SECONDS = 3

SPECIAL_GESTURES = ["STOP", "CLEAR"]

# =====================
# WINDOW
# =====================
cv2.namedWindow("Gesture LSTM Realtime", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture LSTM Realtime", 800, 600)

# =====================
# MAIN LOOP
# =====================
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    frame_data = np.zeros(126, dtype=np.float32)

    # =====================
    # HAND DETECTION
    # =====================
    if res.multi_hand_landmarks:
        last_hand_time = time.time()

        hand = res.multi_hand_landmarks[0]
        hand_label = res.multi_handedness[0].classification[0].label
        base = 0 if hand_label == "Right" else 63

        for j, p in enumerate(hand.landmark):
            idx = base + j * 3
            frame_data[idx:idx+3] = (p.x, p.y, p.z)

        draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
    else:
        current_word = None
        stable_count = 0

    # =====================
    # AUTO CLEAR (NO HAND)
    # =====================
    if time.time() - last_hand_time > AUTO_CLEAR_SECONDS:
        display_word = ""
        sentence.clear()

    # =====================
    # BUFFER
    # =====================
    sequence.append(frame_data)
    if len(sequence) > 30:
        sequence.pop(0)

    # =====================
    # PREDICT
    # =====================
    if len(sequence) == 30:
        try:
            input_data = np.stack(sequence)[None, ...]
        except:
            sequence.clear()
            continue

        probs = model.predict(input_data, verbose=0)[0]

        # ===== TOP-2 FILTER =====
        top2 = np.argsort(probs)[-2:]
        best, second = top2[1], top2[0]

        word = labels[best]
        conf = probs[best]
        margin = conf - probs[second]

        # ===== IGNORE NONE =====
        if word == "NONE":
            current_word = None
            stable_count = 0
            continue

        # ===== RAGU → DIAM =====
        if conf < CONFIDENCE_THRESHOLD or margin < MIN_MARGIN:
            current_word = None
            stable_count = 0
            continue

        # ===== STABILITY CHECK =====
        if word == current_word:
            stable_count += 1
        else:
            current_word = word
            stable_count = 1

        required_stable = SPECIAL_STABLE if word in SPECIAL_GESTURES else STABLE_FRAMES

        if stable_count >= required_stable:
            # =====================
            # ACTION
            # =====================
            if word == "CLEAR":
                display_word = ""
                sentence.clear()

            elif word == "STOP":
                if sentence:
                    speak(" ".join(sentence))
                sentence.clear()

            else:
                sentence.append(word)
                display_word = " ".join(sentence)

                if display_word != last_spoken_word:
                    speak(word)
                    last_spoken_word = display_word

            # RESET
            sequence.clear()
            stable_count = 0
            current_word = None

        # DEBUG TEXT
        cv2.putText(
            frame,
            f"{word} ({conf:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # =====================
    # DISPLAY TEXT
    # =====================
    cv2.putText(
        frame,
        display_word,
        (20, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 0),
        3
    )

    cv2.imshow("Gesture LSTM Realtime", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
