"""
collect_data.py
---------------
Webcam-based data collection for ISL gesture recognition.
Uses MediaPipe Holistic to extract landmarks per frame and saves sequences as .npy files.

Usage:
    python src/collect_data.py
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import time

# ── Configuration ────────────────────────────────────────────────────────────
GESTURES = [
    "Accident", "Allergy", "Doctor", "Hello", "Help",
    "Love", "Money", "Pain", "Police", "Thank_you", "Wait"
]
SEQUENCE_LENGTH = 30        # Frames per gesture sequence
NUM_SEQUENCES   = 30        # Sequences per gesture
DATA_DIR        = os.path.join(os.path.dirname(__file__), "..", "data", "landmarks")

# ── MediaPipe Setup ───────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles


def extract_keypoints(results):
    """
    Flatten all landmark groups from MediaPipe Holistic into a single 1D array.
    Missing landmarks are zero-padded.

    Feature layout (total varies slightly by version, ~1662 values):
        pose       : 33 landmarks × 4 (x, y, z, visibility) = 132
        face       : 468 landmarks × 3 (x, y, z)            = 1404
        left_hand  : 21 landmarks × 3 (x, y, z)             = 63
        right_hand : 21 landmarks × 3 (x, y, z)             = 63
    Total: 1662
    """
    pose = (
        np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 4)
    )
    face = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks else np.zeros(468 * 3)
    )
    lh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )
    rh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])


def mediapipe_detection(frame, model):
    """Convert BGR frame → RGB → process → return results."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = model.process(frame_rgb)
    frame_rgb.flags.writeable = True
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), results


def draw_landmarks(frame, results):
    """Draw all landmark overlays on the frame."""
    # Face mesh (subtle)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
    # Pose (body)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
    # Hands
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )


def create_directories():
    """Create output directories for all gestures."""
    for gesture in GESTURES:
        for seq_idx in range(NUM_SEQUENCES):
            path = os.path.join(DATA_DIR, gesture, str(seq_idx))
            os.makedirs(path, exist_ok=True)
    print(f"[INFO] Data directories ready at: {os.path.abspath(DATA_DIR)}")


def overlay_text(frame, text, pos, color=(255, 255, 255), scale=1, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def collect():
    create_directories()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for i, gesture in enumerate(GESTURES):
            print(f"\n[COLLECTING] Gesture: '{gesture}'")

            # ── 10-second prep delay between gestures ──────────────────
            if i > 0:
                print(f"  [PREPARE] 10 seconds to rest before starting '{gesture}'...")
                for rest_countdown in range(10, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    overlay_text(frame, f"Next Gesture: {gesture}", (w//2 - 180, h//2 - 50), (0, 255, 120), 1.5, 3)
                    overlay_text(frame, f"Resting... {rest_countdown}s remaining", (w//2 - 250, h//2 + 50), (0, 200, 255), 1.5, 3)
                    cv2.imshow("ISL Data Collection", frame)
                    if cv2.waitKey(1000) & 0xFF == ord('q'):
                        print("[QUIT] Collection interrupted by user during rest.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

            for seq_idx in range(NUM_SEQUENCES):
                # ── Countdown before each sequence ───────────────────────────
                for countdown in range(3, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                    overlay_text(frame, f"Gesture: {gesture}", (30, 60), (0, 255, 120), 1.2, 2)
                    overlay_text(frame, f"Sequence {seq_idx + 1}/{NUM_SEQUENCES}", (30, 100), (200, 200, 200), 0.8, 1)
                    overlay_text(frame, f"Starting in {countdown}...", (w//2 - 120, h//2), (0, 200, 255), 2, 3)
                    cv2.imshow("ISL Data Collection", frame)
                    cv2.waitKey(1000)

                # ── Record SEQUENCE_LENGTH frames ─────────────────────────────
                for frame_idx in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    frame, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(frame, results)

                    h, w = frame.shape[:2]
                    # HUD
                    cv2.rectangle(frame, (0, 0), (w, 50), (57, 22, 169), -1)
                    overlay_text(frame, f"RECORDING  {gesture}  [{seq_idx+1}/{NUM_SEQUENCES}]",
                                 (10, 35), (255, 255, 255), 0.8, 2)
                    # Progress bar
                    bar_w = int((frame_idx / SEQUENCE_LENGTH) * (w - 20))
                    cv2.rectangle(frame, (10, h - 20), (w - 10, h - 5), (50, 50, 50), -1)
                    cv2.rectangle(frame, (10, h - 20), (10 + bar_w, h - 5), (0, 255, 120), -1)

                    cv2.imshow("ISL Data Collection", frame)
                    cv2.waitKey(1) & 0xFF

                    # Extract and save keypoints
                    keypoints = extract_keypoints(results)
                    save_path = os.path.join(DATA_DIR, gesture, str(seq_idx), f"{frame_idx}.npy")
                    np.save(save_path, keypoints)

                print(f"  [OK] Sequence {seq_idx + 1:02d}/{NUM_SEQUENCES} saved  ({SEQUENCE_LENGTH} frames)")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[QUIT] Collection interrupted by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            print(f"[DONE] All sequences for '{gesture}' collected.")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[COMPLETE] Data collection finished for all gestures!")


if __name__ == "__main__":
    collect()
