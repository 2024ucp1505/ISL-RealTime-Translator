"""
preprocess.py
-------------
Loads raw .npy landmark files, applies normalization + augmentation,
and builds X (sequences) and y (labels) arrays ready for training.

Usage:
    python src/preprocess.py
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import json

# ── Configuration ─────────────────────────────────────────────────────────────
GESTURES        = [
    "Accident", "Allergy", "Doctor", "Hello", "Help",
    "Love", "Money", "Pain", "Police", "Thank_you", "Wait"
]
SEQUENCE_LENGTH = 30
DATA_DIR        = os.path.join(os.path.dirname(__file__), "..", "data", "landmarks")
MODELS_DIR      = os.path.join(os.path.dirname(__file__), "..", "models")

# Feature sizes (must match collect_data.py)
POSE_SIZE       = 33 * 4    # 132
FACE_SIZE       = 468 * 3   # 1404
HAND_SIZE       = 21 * 3    # 63 per hand
FEATURE_SIZE    = POSE_SIZE + FACE_SIZE + HAND_SIZE * 2  # 1662


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_sequence(sequence):
    """
    Position-invariant normalization:
    - Hands: subtract wrist landmark so only shape/movement matters
    - Body:  subtract hip midpoint
    - Face:  subtract nose tip

    Args:
        sequence: np.array of shape (30, 1662)
    Returns:
        normalized: np.array of shape (30, 1662)
    """
    normalized = sequence.copy()

    for i, frame in enumerate(sequence):
        # ── Pose: hip-relative ───────────────────────────────────────────────
        # pose landmarks: index 0-131, each lm = (x, y, z, vis) → 4 values
        # left hip = lm 23, right hip = lm 24
        left_hip  = frame[23*4 : 23*4+3]   # x,y,z only
        right_hip = frame[24*4 : 24*4+3]
        hip_mid   = (left_hip + right_hip) / 2.0

        pose_slice = frame[:POSE_SIZE].reshape(33, 4)
        pose_slice[:, :3] -= hip_mid          # subtract from x,y,z (not visibility)
        normalized[i, :POSE_SIZE] = pose_slice.flatten()

        # ── Face: nose-relative ──────────────────────────────────────────────
        # face landmarks: index POSE_SIZE onwards, nose tip ≈ lm 1 in face mesh
        face_start = POSE_SIZE
        face_slice = frame[face_start : face_start + FACE_SIZE].reshape(468, 3)
        nose_tip   = face_slice[1].copy()
        face_slice -= nose_tip
        normalized[i, face_start : face_start + FACE_SIZE] = face_slice.flatten()

        # ── Left hand: wrist-relative ─────────────────────────────────────────
        lh_start = POSE_SIZE + FACE_SIZE
        lh_slice = frame[lh_start : lh_start + HAND_SIZE].reshape(21, 3)
        if np.any(lh_slice != 0):             # only if hand was detected
            wrist = lh_slice[0].copy()
            lh_slice -= wrist
        normalized[i, lh_start : lh_start + HAND_SIZE] = lh_slice.flatten()

        # ── Right hand: wrist-relative ────────────────────────────────────────
        rh_start = POSE_SIZE + FACE_SIZE + HAND_SIZE
        rh_slice = frame[rh_start : rh_start + HAND_SIZE].reshape(21, 3)
        if np.any(rh_slice != 0):
            wrist = rh_slice[0].copy()
            rh_slice -= wrist
        normalized[i, rh_start : rh_start + HAND_SIZE] = rh_slice.flatten()

    return normalized


# ── Data Augmentation ─────────────────────────────────────────────────────────

def augment_sequence(sequence, seed=None):
    """
    Apply random augmentations to a single sequence.
    Returns a list with the original + augmented versions.
    """
    rng = np.random.default_rng(seed)
    augmented = [sequence]

    # 1. Temporal jitter: shift frames ±2
    shift = rng.integers(-2, 3)
    jittered = np.roll(sequence, shift, axis=0)
    augmented.append(jittered)

    # 2. Gaussian noise injection
    noise = rng.normal(0, 0.001, sequence.shape)
    augmented.append(sequence + noise)

    # 3. Horizontal flip (mirror x-coordinates)
    flipped = sequence.copy()
    # Swap left and right hand sections
    lh_start = POSE_SIZE + FACE_SIZE
    rh_start = lh_start + HAND_SIZE
    lh = flipped[:, lh_start : lh_start + HAND_SIZE].copy()
    rh = flipped[:, rh_start : rh_start + HAND_SIZE].copy()
    flipped[:, lh_start : lh_start + HAND_SIZE] = rh
    flipped[:, rh_start : rh_start + HAND_SIZE] = lh
    # Flip x coordinates (negate x for all features)
    # Rough approximation: negate every 3rd value (x) for hand features
    for start in [POSE_SIZE + FACE_SIZE, POSE_SIZE + FACE_SIZE + HAND_SIZE]:
        section = flipped[:, start : start + HAND_SIZE].reshape(SEQUENCE_LENGTH, 21, 3)
        section[:, :, 0] *= -1          # negate x
        flipped[:, start : start + HAND_SIZE] = section.reshape(SEQUENCE_LENGTH, HAND_SIZE)
    augmented.append(flipped)

    return augmented


# ── Dataset Builder ───────────────────────────────────────────────────────────

def load_dataset(augment=True):
    """
    Load all sequences from DATA_DIR, normalize, optionally augment,
    and return (X, y, label_map).

    Returns:
        X: np.array of shape (N, SEQUENCE_LENGTH, FEATURE_SIZE)
        y: np.array of shape (N, num_classes) — one-hot encoded
        label_map: dict {index: gesture_name}
    """
    label_map = {i: g for i, g in enumerate(GESTURES)}
    X_list, y_list = [], []

    for label_idx, gesture in enumerate(GESTURES):
        gesture_path = os.path.join(DATA_DIR, gesture)
        if not os.path.exists(gesture_path):
            print(f"[WARNING] Missing data for gesture '{gesture}' — skipping.")
            continue

        sequences_found = 0
        for seq_idx in os.listdir(gesture_path):
            seq_path = os.path.join(gesture_path, seq_idx)
            frames = []
            for frame_idx in range(SEQUENCE_LENGTH):
                frame_file = os.path.join(seq_path, f"{frame_idx}.npy")
                if not os.path.exists(frame_file):
                    frames.append(np.zeros(FEATURE_SIZE))
                else:
                    frames.append(np.load(frame_file))

            sequence = np.array(frames)            # (30, 1662)
            sequence = normalize_sequence(sequence)

            if augment:
                for aug_seq in augment_sequence(sequence):
                    X_list.append(aug_seq)
                    y_list.append(label_idx)
            else:
                X_list.append(sequence)
                y_list.append(label_idx)

            sequences_found += 1

        print(f"  [OK] {gesture}: {sequences_found} sequences loaded"
              + (f" → {sequences_found * 4} after augmentation" if augment else ""))

    X = np.array(X_list)       # (N, 30, 1662)
    y = np.array(y_list)       # (N,)

    num_classes = len(GESTURES)
    y_onehot = to_categorical(y, num_classes=num_classes)

    return X, y_onehot, label_map


def augment_dataset(X, y):
    """
    Safely augment a dataset AFTER splitting.
    X shape: (N, 30, 1662)
    y shape: (N, num_classes)
    """
    X_aug, y_aug = [], []
    for i in range(len(X)):
        for aug_seq in augment_sequence(X[i]):
            X_aug.append(aug_seq)
            y_aug.append(y[i])
    return np.array(X_aug), np.array(y_aug)


def save_label_map(label_map):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "label_map.json")
    # Convert int keys to str for JSON serialization
    json.dump({str(k): v for k, v in label_map.items()}, open(path, "w"))
    print(f"[INFO] Label map saved → {path}")


def get_splits(X, y, val_size=0.15, test_size=0.15, random_state=42):
    """Return (X_train, X_val, X_test, y_train, y_val, y_test)."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y.argmax(axis=1)
    )
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state,
        stratify=y_temp.argmax(axis=1)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    print("[INFO] Loading dataset...")
    X, y, label_map = load_dataset(augment=True)
    print(f"\n[SUMMARY]")
    print(f"  Total samples  : {X.shape[0]}")
    print(f"  Sequence shape : {X.shape[1:]}")
    print(f"  Classes        : {len(GESTURES)}")
    save_label_map(label_map)

    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(X, y)
    print(f"  Train : {X_train.shape[0]}")
    print(f"  Val   : {X_val.shape[0]}")
    print(f"  Test  : {X_test.shape[0]}")
