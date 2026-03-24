"""
evaluate.py
-----------
Loads the trained model and evaluates on the held-out test set.
Generates all metrics from the research paper:
  - Accuracy, MAE, MSE, R² Score
  - F1 Score (per class + weighted)
  - Confusion matrix heatmap
  - Comparison table (vs. paper's benchmarks)

Usage:
    python src/evaluate.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, r2_score
)

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_dataset, get_splits, GESTURES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
EVAL_DIR   = os.path.join(MODELS_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)


def load_model():
    model_path = os.path.join(MODELS_DIR, "isl_lstm_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    print(f"[INFO] Loading model from {model_path}")
    return tf.keras.models.load_model(model_path)


def compute_all_metrics(model, X_test, y_test):
    """Compute all metrics matching the research paper."""
    y_pred_probs = model.predict(X_test, verbose=0)           # (N, num_classes)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = np.argmax(y_test, axis=1)

    # Standard metrics
    acc = np.mean(y_pred == y_true)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    f1_per_class  = f1_score(y_true, y_pred, average=None, labels=list(range(len(GESTURES))))
    f1_weighted   = f1_score(y_true, y_pred, average='weighted')

    return {
        "accuracy"    : acc,
        "mae"         : mae,
        "mse"         : mse,
        "r2_score"    : r2,
        "f1_weighted" : f1_weighted,
        "f1_per_class": f1_per_class.tolist(),
        "y_true"      : y_true,
        "y_pred"      : y_pred,
    }


def print_report(metrics):
    """Pretty-print all metrics."""
    print("\n" + "═" * 55)
    print("  ISL LSTM Model — Evaluation Report")
    print("═" * 55)
    print(f"  Accuracy   : {metrics['accuracy'] * 100:.2f}%    (Paper: 96.97%)")
    print(f"  MAE        : {metrics['mae']:.4f}         (Paper: 0.0303)")
    print(f"  MSE        : {metrics['mse']:.4f}         (Paper: 0.0303)")
    print(f"  R² Score   : {metrics['r2_score']:.4f}         (Paper: 0.9969)")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    print("─" * 55)
    print("  Per-Class F1 Scores:")
    for i, gesture in enumerate(GESTURES):
        f1 = metrics['f1_per_class'][i]
        bar = "█" * int(f1 * 20)
        print(f"  {gesture:<12s}: {f1:.4f}  {bar}")
    print("═" * 55)

    # Comparison table
    print("\n  ┌─────────────────────────┬───────────┐")
    print(  "  │ Model                   │ Accuracy  │")
    print(  "  ├─────────────────────────┼───────────┤")
    baselines = [("SVM",        "~85.00%"),
                 ("k-NN",       "~80.00%"),
                 ("Basic RNN",  "82.50%"),
                 ("CNN",        "93.50%"),
                 ("Paper LSTM", "96.97%")]
    for name, acc_str in baselines:
        print(f"  │ {name:<23s} │ {acc_str:<9s} │")
    ours = f"{metrics['accuracy'] * 100:.2f}%"
    print(  "  ├─────────────────────────┼───────────┤")
    print(f"  │ {'Our LSTM (this system)':<23s} │ {ours:<9s} │")
    print(  "  └─────────────────────────┴───────────┘\n")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=GESTURES, yticklabels=GESTURES)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].tick_params(axis='x', rotation=45)

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens', ax=axes[1],
                xticklabels=GESTURES, yticklabels=GESTURES, vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    save_path = os.path.join(EVAL_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix saved → {save_path}")


def plot_f1_bars(metrics):
    f1_scores  = metrics['f1_per_class']
    colors     = ['#2ecc71' if f1 >= 0.9 else '#e67e22' if f1 >= 0.7 else '#e74c3c'
                  for f1 in f1_scores]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(GESTURES, f1_scores, color=colors, edgecolor='white', linewidth=0.5)

    # Value labels
    for bar, f1 in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=9)

    ax.axhline(0.9, color='gray', linestyle='--', linewidth=1, label='0.90 threshold')
    ax.set_ylim(0, 1.15)
    ax.set_xlabel('Gesture', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Scores', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(EVAL_DIR, "f1_scores.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] F1 score chart saved → {save_path}")


def main():
    model = load_model()

    print("[INFO] Loading test data...")
    X, y, _ = load_dataset(augment=False)
    _, _, X_test, _, _, y_test = get_splits(X, y)
    print(f"  Test set size: {X_test.shape[0]} samples")

    metrics = compute_all_metrics(model, X_test, y_test)
    print_report(metrics)

    plot_confusion_matrix(metrics['y_true'], metrics['y_pred'])
    plot_f1_bars(metrics)

    # Save JSON summary
    summary = {k: v for k, v in metrics.items() if k not in ('y_true', 'y_pred')}
    json.dump(summary, open(os.path.join(EVAL_DIR, "metrics_summary.json"), "w"), indent=2)
    print(f"[INFO] Metrics summary saved → {EVAL_DIR}/metrics_summary.json")


if __name__ == "__main__":
    main()
