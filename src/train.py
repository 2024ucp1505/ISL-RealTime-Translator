"""
train.py
--------
Trains the Sequential LSTM model on the ISL dataset.
Architecture follows the research paper with added Dropout + ReduceLR callbacks.

Usage:
    python src/train.py
"""

import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress verbose TF logs

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)

# Add src to path so we can import preprocess
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_dataset, augment_dataset, save_label_map, get_splits, GESTURES, SEQUENCE_LENGTH, FEATURE_SIZE

# ── Configuration ─────────────────────────────────────────────────────────────
MODELS_DIR     = os.path.join(os.path.dirname(__file__), "..", "models")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")
LOG_DIR        = os.path.join(MODELS_DIR, "logs")
BATCH_SIZE     = 32
MAX_EPOCHS     = 200
LEARNING_RATE  = 0.0001      # From paper
NUM_CLASSES    = len(GESTURES)


def build_model(input_shape, num_classes):
    """
    Sequential LSTM model as described in the paper:
    - 2 × LSTM layers (32 units, SELU activation)
    - Dense output with Softmax
    Additions over paper:
    - Dropout(0.2) between LSTM layers and before output
    - Extra Dense(64) hidden layer for capacity
    """
    model = Sequential([
        LSTM(32, return_sequences=True, activation='selu', input_shape=input_shape,
             kernel_initializer='lecun_normal',   # needed for SELU self-normalization
             recurrent_dropout=0.0),
        Dropout(0.2),

        LSTM(32, return_sequences=False, activation='selu',
             kernel_initializer='lecun_normal'),
        Dropout(0.2),

        Dense(64, activation='relu'),
        Dropout(0.1),

        Dense(num_classes, activation='softmax')
    ], name="ISL_LSTM")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks(checkpoint_path, log_dir):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return [
        EarlyStopping(
            monitor='val_loss', patience=20,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10,
            min_lr=1e-6, verbose=1
        ),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]


def plot_history(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'],    label='Train Accuracy',  color='#4A90D9')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   color='#E74C3C')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'],     label='Train Loss',  color='#4A90D9')
    axes[1].plot(history.history['val_loss'], label='Val Loss',   color='#E74C3C')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Training history plot saved → {save_path}")


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    print("[INFO] Loading dataset (preventing data leakage)...")
    X, y, label_map = load_dataset(augment=False)
    save_label_map(label_map)

    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(X, y)
    
    print(f"[INFO] Augmenting training set ONLY...")
    X_train, y_train = augment_dataset(X_train, y_train)
    
    print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print(f"  Input shape: {X_train.shape[1:]}")

    # ── Build model ───────────────────────────────────────────────────────────
    input_shape = (SEQUENCE_LENGTH, FEATURE_SIZE)
    model = build_model(input_shape, NUM_CLASSES)
    model.summary()

    # ── Train ─────────────────────────────────────────────────────────────────
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.h5")
    eval_dir        = os.path.join(MODELS_DIR, "evaluation")
    callbacks       = get_callbacks(checkpoint_path, LOG_DIR)

    print(f"\n[INFO] Training for up to {MAX_EPOCHS} epochs (early stopping active)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[INFO] Evaluating on held-out test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  Test Loss     : {test_loss:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    final_path = os.path.join(MODELS_DIR, "isl_lstm_model.h5")
    model.save(final_path)
    print(f"[INFO] Final model saved → {final_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_history(history, eval_dir)

    # ── Save test metrics for evaluate.py ────────────────────────────────────
    metrics = {"test_accuracy": float(test_acc), "test_loss": float(test_loss)}
    json.dump(metrics, open(os.path.join(MODELS_DIR, "test_metrics.json"), "w"), indent=2)
    print("[INFO] Test metrics saved → models/test_metrics.json")

    print(f"\n✅ Training complete!  Test Accuracy: {test_acc * 100:.2f}%")
    print(f"   Paper baseline: 96.97%  |  {'✅ Matched/Exceeded!' if test_acc >= 0.9697 else '🔧 Below paper baseline — consider more data.'}")


if __name__ == "__main__":
    main()
