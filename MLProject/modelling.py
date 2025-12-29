import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix
)

def load_processed(preproc_dir: Path):
    train_path = preproc_dir / "customer_churn_train.csv"
    test_path  = preproc_dir / "customer_churn_test.csv"
    target_path = preproc_dir / "target_name.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"Train file tidak ditemukan: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file tidak ditemukan: {test_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"target_name.txt tidak ditemukan: {target_path}")

    target_col = target_path.read_text(encoding="utf-8").strip()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Target '{target_col}' tidak ada di train/test.")

    X_train = train_df.drop(columns=[target_col]).to_numpy()
    y_train = train_df[target_col].to_numpy()
    X_test  = test_df.drop(columns=[target_col]).to_numpy()
    y_test  = test_df[target_col].to_numpy()

    return X_train, y_train, X_test, y_test, target_col

def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_proba))
        except Exception:
            pass
        try:
            proba_pos = y_proba[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
        except Exception:
            pass
    return metrics

def save_confusion_matrix(cm, out_path: Path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, aspect="auto")
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc_dir", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=3.0)
    ap.add_argument("--max_iter", type=int, default=1000)
    args = ap.parse_args()

    preproc_dir = Path(args.preproc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, target_col = load_processed(preproc_dir)

    model = LogisticRegression(
        C=args.C,
        penalty="l2",
        solver="liblinear",
        max_iter=args.max_iter,
        random_state=args.seed
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = compute_metrics(y_test, y_pred, y_proba=y_proba)

    # Save model + metrics
    dump(model, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Confusion matrix artifact
    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm, out_dir / "confusion_matrix.png")

    # Metadata
    meta = {
        "target_col": target_col,
        "params": {"C": args.C, "max_iter": args.max_iter, "seed": args.seed},
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("DONE. Saved artifacts to:", out_dir.resolve())
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()