"""
infer_realtime.py
-----------------
Real-time ISL gesture recognition from webcam.
Uses the trained LSTM model + MediaPipe Holistic.

Usage:
    python src/infer_realtime.py

Controls:
    q  →  Quit
    c  →  Clear the sentence buffer
    s  →  Save current sentence to output.txt
"""

import cv2
import numpy as np
import json
import os
import sys
import time
from collections import deque

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import mediapipe as mp
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import normalize_sequence, SEQUENCE_LENGTH, FEATURE_SIZE

# Re-use the extract function from preprocess (defined inline below for standalone use)
# ── Re-implement extract_keypoints locally so this file can run standalone ────
POSE_SIZE = 33 * 4
FACE_SIZE = 468 * 3
HAND_SIZE = 21 * 3


def extract_keypoints(results):
    pose = (
        np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(POSE_SIZE)
    )
    face = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks else np.zeros(FACE_SIZE)
    )
    lh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(HAND_SIZE)
    )
    rh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(HAND_SIZE)
    )
    return np.concatenate([pose, face, lh, rh])


# ── Configuration ─────────────────────────────────────────────────────────────
MODELS_DIR         = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH         = os.path.join(MODELS_DIR, "isl_lstm_model.h5")
LABEL_MAP_PATH     = os.path.join(MODELS_DIR, "label_map.json")
CONFIDENCE_THRESH  = 0.70    # Only show prediction if confidence ≥ 70%
SENTENCE_COOLDOWN  = 1.5     # Seconds between adding the same word twice

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
CLR_BG_PANEL  = (30, 20, 60)
CLR_ACCENT    = (180, 120, 0)
CLR_GREEN     = (50, 200, 80)
CLR_RED       = (60, 60, 220)
CLR_WHITE     = (255, 255, 255)
CLR_GRAY      = (150, 150, 150)


def load_label_map():
    with open(LABEL_MAP_PATH) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def mediapipe_detection(frame, model):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = model.process(rgb)
    rgb.flags.writeable = True
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), results


def draw_landmarks(frame, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    spec_thin  = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    spec_hand  = mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3)
    spec_hand2 = mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)

    if results.face_landmarks:
        mp_drawing.draw_landmarks(frame, results.face_landmarks,
                                  mp_holistic.FACEMESH_CONTOURS, spec_thin, spec_thin)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                  mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS, spec_hand, spec_hand2)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS, spec_hand, spec_hand2)


def draw_hud(frame, gesture, confidence, sentence, buffer_len, fps):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Top bar ──────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 60), CLR_BG_PANEL, -1)
    fps_text = f"FPS: {fps:.0f}"
    cv2.putText(frame, "ISL Real-Time Recognition", (12, 40), font, 0.9, CLR_WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (w - 100, 40), font, 0.7, CLR_GRAY, 1, cv2.LINE_AA)

    # ── Buffer progress bar ──────────────────────────────────────────────────
    bar_x, bar_y, bar_h = 10, 65, 8
    bar_w_total = w - 20
    bar_fill    = int((buffer_len / SEQUENCE_LENGTH) * bar_w_total)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_total, bar_y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fill, bar_y + bar_h), CLR_GREEN, -1)

    # ── Predicted gesture (large, center) ────────────────────────────────────
    if gesture and confidence >= CONFIDENCE_THRESH:
        label_text = gesture.replace("_", " ").upper()
        conf_text  = f"{confidence * 100:.1f}%"

        # Gesture label
        (tw, th), _ = cv2.getTextSize(label_text, font, 2.0, 3)
        cx = (w - tw) // 2
        cv2.putText(frame, label_text, (cx, h // 2 - 10), font, 2.0, CLR_GREEN, 3, cv2.LINE_AA)

        # Confidence
        (cw, _), _ = cv2.getTextSize(conf_text, font, 0.9, 2)
        cv2.putText(frame, conf_text, ((w - cw) // 2, h // 2 + 35), font, 0.9, CLR_ACCENT, 2, cv2.LINE_AA)

        # Confidence rectangle bar
        bar_len = int(confidence * 200)
        cx2     = (w - 200) // 2
        cv2.rectangle(frame, (cx2, h // 2 + 50), (cx2 + 200, h // 2 + 65), (50, 50, 50), -1)
        cv2.rectangle(frame, (cx2, h // 2 + 50), (cx2 + bar_len, h // 2 + 65), CLR_GREEN, -1)

    # ── Sentence panel (bottom) ──────────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 60), (w, h), CLR_BG_PANEL, -1)
    sentence_display = " ".join(sentence) if sentence else "Start signing..."
    # Truncate if too long
    if len(sentence_display) > 60:
        sentence_display = "..." + sentence_display[-57:]
    cv2.putText(frame, sentence_display, (12, h - 20), font, 0.75, CLR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "[C] Clear  [S] Save  [Q] Quit",
                (w - 280, h - 5), font, 0.45, CLR_GRAY, 1, cv2.LINE_AA)

    return frame


def main():
    # ── Load model + labels ──────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        print("  Please run: python src/train.py first.")
        return

    print("[INFO] Loading model...")
    model     = tf.keras.models.load_model(MODEL_PATH)
    label_map = load_label_map()
    print(f"[INFO] Model loaded. {len(label_map)} gestures: {list(label_map.values())}")

    # ── Webcam ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_holistic = mp.solutions.holistic
    frame_buffer   = deque(maxlen=SEQUENCE_LENGTH)   # Rolling 30-frame window
    sentence       = []                               # Accumulated glosses
    last_added     = {"word": None, "time": 0.0}      # Cooldown tracker
    current_pred   = (None, 0.0)                      # (gesture, confidence)
    prev_time      = time.time()

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            now      = time.time()
            fps      = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # ── MediaPipe ────────────────────────────────────────────────────
            frame, results = mediapipe_detection(frame, holistic)
            draw_landmarks(frame, results)

            # ── Feature extraction ───────────────────────────────────────────
            keypoints = extract_keypoints(results)
            frame_buffer.append(keypoints)

            # ── Prediction when buffer is full ───────────────────────────────
            gesture, confidence = current_pred
            if len(frame_buffer) == SEQUENCE_LENGTH:
                sequence = np.array(frame_buffer)          # (30, 1662)
                sequence = normalize_sequence(sequence)    # Apply positional normalization
                input_seq = sequence[np.newaxis, ...]      # (1, 30, 1662)
                probs     = model.predict(input_seq, verbose=0)[0]
                pred_idx  = np.argmax(probs)
                confidence = float(probs[pred_idx])
                gesture    = label_map.get(pred_idx, "Unknown")
                current_pred = (gesture, confidence)

                # ── Sentence builder ──────────────────────────────────────────
                if confidence >= CONFIDENCE_THRESH:
                    elapsed = now - last_added["time"]
                    if gesture != last_added["word"] or elapsed > SENTENCE_COOLDOWN:
                        sentence.append(gesture.replace("_", " "))
                        last_added = {"word": gesture, "time": now}

            # ── Draw HUD ─────────────────────────────────────────────────────
            draw_hud(frame, gesture, confidence, sentence, len(frame_buffer), fps)
            cv2.imshow("ISL Real-Time Recognition", frame)

            # ── Key controls ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence.clear()
                print("[INFO] Sentence cleared.")
            elif key == ord('s'):
                out_path = os.path.join(os.path.dirname(__file__), "..", "output.txt")
                with open(out_path, "a") as f:
                    f.write(" ".join(sentence) + "\n")
                print(f"[INFO] Sentence saved → {out_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")


if __name__ == "__main__":
    main()
