import cv2, mediapipe as mp, numpy as np, os

# =====================
# INPUT LABEL
# =====================
LABEL = input("Masukkan label gesture: ")
SAVE_DIR = f"data/{LABEL}"
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================
# MEDIAPIPE
# =====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
draw = mp.solutions.drawing_utils

# =====================
# IP WEBCAM
# =====================
IP_CAM = "http://192.168.1.4:8080/video"
cap = cv2.VideoCapture(IP_CAM, cv2.CAP_FFMPEG)

# =====================
# WINDOW
# =====================
cv2.namedWindow("Collect Gesture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Collect Gesture", 800, 600)

# =====================
# STATE
# =====================
sequence = []
count = len(os.listdir(SAVE_DIR))

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

    # =====================
    # FRAME DATA (PASTI 126)
    # =====================
    frame_data = np.zeros(126, dtype=np.float32)

    if res.multi_hand_landmarks:
        for i, hand in enumerate(res.multi_hand_landmarks):
            hand_label = res.multi_handedness[i].classification[0].label
            base = 0 if hand_label == "Right" else 63

            for j, p in enumerate(hand.landmark):
                idx = base + j * 3
                frame_data[idx:idx+3] = (p.x, p.y, p.z)

            draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # =====================
    # BUFFER AMAN
    # =====================
    sequence.append(frame_data)

    cv2.putText(
        frame,
        f"Frames: {len(sequence)}/30",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Diam & ulang gesture (ESC keluar)",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Collect Gesture", frame)

    # =====================
    # SAVE SEQUENCE
    # =====================
    if len(sequence) == 30:
        np.save(
            f"{SAVE_DIR}/{count}.npy",
            np.stack(sequence)
        )
        print(f"✅ Saved: {LABEL} sample {count}")
        sequence.clear()
        count += 1

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
