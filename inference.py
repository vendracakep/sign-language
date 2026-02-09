import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
from gtts import gTTS
from playsound import playsound
import os
import time
from collections import deque

# ============================================
# KONFIGURASI UI/UX
# ============================================
WINDOW_NAME = "Sign Language Recognition - Vendra"
CAMERA_WIDTH = 1280  # Lebih besar dari 320!
CAMERA_HEIGHT = 720  # Lebih besar dari 240!

# Color Palette (Modern & Professional)
COLOR_PRIMARY = (255, 107, 107)      # Coral Red
COLOR_SECONDARY = (78, 205, 196)     # Turquoise
COLOR_ACCENT = (255, 193, 7)         # Amber
COLOR_SUCCESS = (102, 255, 102)      # Light Green
COLOR_TEXT_LIGHT = (255, 255, 255)   # White
COLOR_TEXT_DARK = (50, 50, 50)       # Dark Gray
COLOR_OVERLAY = (30, 30, 30)         # Dark overlay

# ============================================
# LOAD MODEL
# ============================================
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# ============================================
# SETUP KAMERA
# ============================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

# ============================================
# MEDIAPIPE HANDS SETUP
# ============================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,  
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Custom Drawing Specs untuk Hand Landmarks
landmark_drawing_spec = mp_drawing.DrawingSpec(
    color=(0, 255, 255),  # Cyan untuk landmark points
    thickness=3,
    circle_radius=4
)

connection_drawing_spec = mp_drawing.DrawingSpec(
    color=COLOR_SECONDARY,  # Turquoise untuk connections
    thickness=3
)

# ============================================
# LABELS & TTS CACHE
# ============================================
labels_dict = {
    0: 'Halo', 
    1: 'Perkenalkan', 
    2: 'Nama Saya', 
    3: 'Vendra', 
    4: 'Terima Kasih'
}

TTS_CACHE = {label: f"tts_cache_{label.replace(' ', '_')}.mp3" for label in labels_dict.values()}
for label, file in TTS_CACHE.items():
    if not os.path.exists(file):
        tts = gTTS(text=label, lang='id')
        tts.save(file)

# ============================================
# VARIABEL STATE
# ============================================
last_prediction = None
last_speak_time = 0
SPEAK_DELAY = 1.0  # detik minimal antar suara (sama seperti versi lama)

# Prediction history untuk smoothing
prediction_history = deque(maxlen=5)

# FPS tracking
fps_history = deque(maxlen=30)
prev_time = time.time()

# ============================================
# FUNGSI TTS ASYNC
# ============================================
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

# ============================================
# FUNGSI UI HELPER
# ============================================
def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius=20):
    """Draw rounded rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw main rectangle
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    
    # Draw corners
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)

def draw_text_with_background(img, text, position, font_scale=1.0, 
                               text_color=COLOR_TEXT_LIGHT, 
                               bg_color=COLOR_OVERLAY,
                               padding=10):
    """Draw text with background"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    
    # Draw background
    draw_rounded_rectangle(
        img,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        bg_color,
        -1,
        radius=10
    )
    
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    return text_height + baseline + 2 * padding

def draw_confidence_bar(img, confidence, position, width=200, height=20):
    """Draw confidence progress bar"""
    x, y = position
    
    # Background
    cv2.rectangle(img, (x, y), (x + width, y + height), (50, 50, 50), -1)
    
    # Fill based on confidence
    fill_width = int(width * confidence)
    
    # Color gradient based on confidence
    if confidence > 0.8:
        color = COLOR_SUCCESS
    elif confidence > 0.5:
        color = COLOR_ACCENT
    else:
        color = COLOR_PRIMARY
    
    cv2.rectangle(img, (x, y), (x + fill_width, y + height), color, -1)
    
    # Border
    cv2.rectangle(img, (x, y), (x + width, y + height), COLOR_TEXT_LIGHT, 2)
    
    # Percentage text
    percentage_text = f"{int(confidence * 100)}%"
    cv2.putText(img, percentage_text, (x + width + 10, y + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_LIGHT, 1, cv2.LINE_AA)

def draw_header(img, fps):
    """Draw header panel"""
    height = 80
    
    # Semi-transparent overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], height), COLOR_OVERLAY, -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    
    # Title
    cv2.putText(img, "Sign Language Recognition", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_PRIMARY, 3, cv2.LINE_AA)
    
    # Subtitle
    cv2.putText(img, "By Vendra", (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_LIGHT, 1, cv2.LINE_AA)
    
    # FPS counter
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(img, fps_text, (img.shape[1] - 150, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ACCENT, 2, cv2.LINE_AA)

def draw_prediction_panel(img, prediction, confidence, history):
    """Draw prediction info panel"""
    panel_width = 350
    panel_x = img.shape[1] - panel_width - 20
    panel_y = 100
    
    # Background panel
    overlay = img.copy()
    draw_rounded_rectangle(overlay, 
                          (panel_x, panel_y), 
                          (panel_x + panel_width, panel_y + 280), 
                          COLOR_OVERLAY, -1, radius=15)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    
    # Border
    cv2.rectangle(img, (panel_x, panel_y), (panel_x + panel_width, panel_y + 280), 
                  COLOR_PRIMARY, 2)
    
    # Panel title
    cv2.putText(img, "DETEKSI", (panel_x + 20, panel_y + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_PRIMARY, 2, cv2.LINE_AA)
    
    # Current prediction
    if prediction:
        cv2.putText(img, prediction, (panel_x + 20, panel_y + 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_SUCCESS, 3, cv2.LINE_AA)
        
        # Confidence bar
        cv2.putText(img, "Confidence:", (panel_x + 20, panel_y + 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_LIGHT, 1, cv2.LINE_AA)
        draw_confidence_bar(img, confidence, (panel_x + 20, panel_y + 145), width=310)
    
    # History
    if history:
        cv2.putText(img, "Riwayat:", (panel_x + 20, panel_y + 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_LIGHT, 1, cv2.LINE_AA)
        
        y_offset = 210
        for i, hist_pred in enumerate(list(history)[-3:]):  # Show last 3
            cv2.putText(img, f"• {hist_pred}", (panel_x + 30, panel_y + y_offset + i * 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

def draw_instructions(img):
    """Draw instructions at bottom"""
    instructions = [
        "ESC - Keluar",
        "F - Fullscreen",
        "R - Reset Riwayat"
    ]
    
    y_start = img.shape[0] - 100
    
    # Background
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y_start), (img.shape[1], img.shape[0]), COLOR_OVERLAY, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Instructions
    x_offset = 20
    for instruction in instructions:
        cv2.putText(img, instruction, (x_offset, y_start + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_LIGHT, 1, cv2.LINE_AA)
        x_offset += 200

# ============================================
# MAIN LOOP
# ============================================
fullscreen = False

print("=" * 60)
print("🚀 Sign Language Recognition Started!")
print("=" * 60)
print("Controls:")
print("  ESC - Keluar")
print("  F   - Toggle Fullscreen")
print("  R   - Reset Riwayat")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time
    fps_history.append(fps)
    avg_fps = np.mean(fps_history)

    # Flip frame untuk mirror effect
    frame = cv2.flip(frame, 1)
    
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_character = None
    confidence = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks dengan style custom
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec,
                connection_drawing_spec
            )

        # Extract features
        x_ = [lm.x for lm in hand_landmarks.landmark]
        y_ = [lm.y for lm in hand_landmarks.landmark]

        data_aux = []
        for i in range(len(hand_landmarks.landmark)):
            data_aux.append(hand_landmarks.landmark[i].x - min(x_))
            data_aux.append(hand_landmarks.landmark[i].y - min(y_))

        # Prediction - OPTIMIZED (single call)
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        
        # Get confidence dari predict_proba hanya jika perlu display
        # Tapi untuk TTS, kita pakai logic simple seperti versi lama
        try:
            prediction_proba = model.predict_proba([np.asarray(data_aux)])
            confidence = np.max(prediction_proba)
        except:
            confidence = 1.0  # Fallback jika model tidak support predict_proba
        
        # Add to history
        prediction_history.append(predicted_character)

        # TTS - LOGIC SAMA PERSIS SEPERTI VERSI LAMA
        if predicted_character != last_prediction:
            print(f"✅ Predicted: {predicted_character}")
            speak_async(TTS_CACHE[predicted_character])
            last_prediction = predicted_character

        # Draw bounding box di sekitar tangan
        x1 = int(min(x_) * W) - 20
        y1 = int(min(y_) * H) - 20
        x2 = int(max(x_) * W) + 20
        y2 = int(max(y_) * H) + 20

        # Bounding box dengan rounded corners
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PRIMARY, 3)
        
        # Label di atas bounding box
        draw_text_with_background(
            frame, 
            predicted_character, 
            (x1, y1 - 10),
            font_scale=1.2,
            text_color=COLOR_TEXT_LIGHT,
            bg_color=COLOR_PRIMARY
        )

    # Draw UI Elements
    draw_header(frame, avg_fps)
    draw_prediction_panel(frame, predicted_character, confidence, prediction_history)
    draw_instructions(frame)

    # Show frame
    cv2.imshow(WINDOW_NAME, frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        print("\n👋 Keluar dari aplikasi...")
        break
    elif key == ord('f') or key == ord('F'):  # Fullscreen toggle
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("🖥️  Fullscreen mode")
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            print("🪟 Windowed mode")
    elif key == ord('r') or key == ord('R'):  # Reset history
        prediction_history.clear()
        last_prediction = None
        print("🔄 Riwayat direset")

cap.release()
cv2.destroyAllWindows()
print("\n✨ Aplikasi ditutup dengan sukses!")