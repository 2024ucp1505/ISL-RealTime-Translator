"""
export_tflite.py
----------------
Converts the trained Keras LSTM model to TFLite format with optional quantization.
Enables deployment on mobile devices, web (WASM), and edge hardware.

Usage:
    python src/export_tflite.py
"""

import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_dataset, get_splits, SEQUENCE_LENGTH, FEATURE_SIZE

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "isl_lstm_model.h5")
TFLITE_PATH = os.path.join(MODELS_DIR, "isl_lstm_model.tflite")
TFLITE_QUANT_PATH = os.path.join(MODELS_DIR, "isl_lstm_model_quantized.tflite")


def get_representative_dataset():
    """Generator for post-training quantization calibration."""
    print("[INFO] Loading calibration data...")
    X, y, _ = load_dataset(augment=False)
    X_train, _, _, _, _, _ = get_splits(X, y)
    # Use 100 random samples for calibration
    indices = np.random.choice(len(X_train), min(100, len(X_train)), replace=False)
    for idx in indices:
        sample = X_train[idx:idx+1].astype(np.float32)
        yield [sample]


def export_float32():
    """Convert to TFLite without quantization (float32 — highest accuracy)."""
    model     = tf.keras.models.load_model(MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
    print(f"[OK] Float32 TFLite model → {TFLITE_PATH}")
    print(f"     Size: {size_mb:.2f} MB")
    return tflite_model


def export_quantized():
    """
    Convert to TFLite with full integer quantization (INT8).
    ~4× smaller, ~2× faster inference — ideal for mobile.
    May have slight accuracy reduction (~0.5-1%).
    """
    model     = tf.keras.models.load_model(MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = get_representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.float32  # keep float input for convenience
    converter.inference_output_type = tf.float32  # keep float output

    tflite_model = converter.convert()

    with open(TFLITE_QUANT_PATH, 'wb') as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(TFLITE_QUANT_PATH) / (1024 * 1024)
    print(f"[OK] Quantized INT8 TFLite model → {TFLITE_QUANT_PATH}")
    print(f"     Size: {size_mb:.2f} MB")
    return tflite_model


def validate_tflite(tflite_model_content, X_test, y_test, model_label="TFLite"):
    """Run inference with TFLite interpreter and compare accuracy."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    for i in range(len(X_test)):
        sample = X_test[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        if np.argmax(output) == np.argmax(y_test[i]):
            correct += 1

    acc = correct / len(X_test)
    print(f"[VALIDATION] {model_label} accuracy on test set: {acc * 100:.2f}%")
    return acc


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Keras model not found at {MODEL_PATH}. Run train.py first.")
        return

    print("=" * 50)
    print("  ISL LSTM → TFLite Export")
    print("=" * 50)

    # Export both variants
    print("\n[1/2] Exporting Float32 model...")
    tflite_f32 = export_float32()

    print("\n[2/2] Exporting Quantized INT8 model...")
    try:
        tflite_q = export_quantized()
    except Exception as e:
        print(f"[WARNING] Quantization failed: {e}")
        print("  The quantized model requires calibration data (run data collection + training first).")
        tflite_q = None

    # Validate on test data
    print("\n[INFO] Validating TFLite models...")
    try:
        X, y, _ = load_dataset(augment=False)
        _, _, X_test, _, _, y_test = get_splits(X, y)

        validate_tflite(tflite_f32, X_test, y_test, "Float32 TFLite")
        if tflite_q:
            validate_tflite(tflite_q, X_test, y_test, "Quantized INT8 TFLite")
    except Exception as e:
        print(f"[WARNING] Validation skipped: {e}")

    print("\n✅ TFLite export complete!")
    print(f"   Float32  : {TFLITE_PATH}")
    if tflite_q:
        print(f"   Quantized: {TFLITE_QUANT_PATH}")
    print("\n  Use isl_lstm_model.tflite for:")
    print("  → Android (TFLite Java/Kotlin API)")
    print("  → Web     (TFLite WASM via @tensorflow/tfjs-tflite)")
    print("  → Edge    (Raspberry Pi, Coral, etc.)")


if __name__ == "__main__":
    main()
