import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
from gtts import gTTS
from playsound import playsound
import os
import time

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,  
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Label gesture
labels_dict = {0: 'Halo', 1: 'perkenalkan', 2: 'nama saya', 3: 'Vendra', 4: 'terima kasih'}

# Cache TTS
TTS_CACHE = {label: f"tts_cache_{label}.mp3" for label in labels_dict.values()}
for label, file in TTS_CACHE.items():
    if not os.path.exists(file):
        tts = gTTS(text=label, lang='id')
        tts.save(file)

# Variabel kontrol
last_prediction = None
last_speak_time = 0
SPEAK_DELAY = 0.5  # detik minimal antar suara

# Fungsi TTS asinkron
def speak_async(file_path):
    def _speak():
        global last_speak_time
        if time.time() - last_speak_time < SPEAK_DELAY:
            return
        last_speak_time = time.time()
        try:
            playsound(file_path)
        except Exception as e:
            print(f"[TTS Error] {e}")
    threading.Thread(target=_speak, daemon=True).start()

# Loop utama
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        x_ = [lm.x for lm in hand_landmarks.landmark]
        y_ = [lm.y for lm in hand_landmarks.landmark]

        data_aux = []
        for i in range(len(hand_landmarks.landmark)):
            data_aux.append(hand_landmarks.landmark[i].x - min(x_))
            data_aux.append(hand_landmarks.landmark[i].y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        if predicted_character != last_prediction:
            print(f"Predicted: {predicted_character}")
            speak_async(TTS_CACHE[predicted_character])
            last_prediction = predicted_character

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
