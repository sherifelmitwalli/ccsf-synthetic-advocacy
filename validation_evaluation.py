"""Account-level leave-one-condition-out evaluation for CCSF independent validation.

The evaluator consumes feature tables created after independent model outputs have
been ingested. It never supplies a score for an unexecuted generator condition.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ("ppl_mean", "burstiness", "commercial_policy_framing", "convergence")


def _float_matrix(rows, columns):
    try:
        return np.array([[float(row[c]) for c in columns] for row in rows], dtype=float)
    except KeyError as exc:
        raise ValueError(f"Feature table is missing required column {exc.args[0]!r}.") from exc


def _bootstrap_ci(y, scores, metric, n_boot=2000, seed=20260604):
    y = np.asarray(y, dtype=int); scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    values = []
    for _ in range(n_boot):
        idx = np.concatenate((rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)))
        values.append(metric(y[idx], scores[idx]))
    lo, hi = np.quantile(values, (0.025, 0.975))
    return [float(lo), float(hi)]


def _classification_metrics(y, probability, threshold=0.5):
    prediction = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = precision_score(y, prediction, zero_division=0)
    recall = recall_score(y, prediction, zero_division=0)
    result = {
        "n": int(len(y)), "threshold": threshold,
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "sensitivity": sensitivity, "specificity": specificity,
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "brier_score": float(brier_score_loss(y, probability)),
        "roc_auc_ci": _bootstrap_ci(y, probability, roc_auc_score),
        "pr_auc_ci": _bootstrap_ci(y, probability, average_precision_score),
    }
    if len(y) >= 10:
        observed, predicted = calibration_curve(y, probability, n_bins=min(5, len(y)), strategy="quantile")
        result["calibration_bins"] = [{"mean_predicted": float(p), "fraction_positive": float(o)}
                                      for p, o in zip(predicted, observed)]
    return result


def _prevalence_scenarios(sensitivity, specificity):
    if sensitivity is None or specificity is None:
        return {}
    result = {}
    for prevalence in (0.01, 0.05, 0.10):
        ppv = sensitivity * prevalence / (sensitivity * prevalence + (1 - specificity) * (1 - prevalence))
        result[str(prevalence)] = {"ppv": float(ppv), "false_discovery_rate": float(1 - ppv)}
    return result


def evaluate(feature_csv: str | Path, split_manifest: str | Path, output: str | Path) -> dict:
    with Path(feature_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Feature table is empty.")
    accounts = {row["account_id"] for row in rows}
    if len(accounts) != len(rows):
        raise ValueError("Feature table must have one already-aggregated row per account.")
    manifest = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    result = {"unit": "account", "feature_columns": list(FEATURE_COLUMNS),
              "holdout_field": manifest["field"], "splits": []}
    for index, split in enumerate(manifest["splits"]):
        train_ids, test_ids = set(split["train_account_ids"]), set(split["test_account_ids"])
        if train_ids & test_ids:
            raise ValueError("Split has overlapping account identifiers.")
        train = [row for row in rows if row["account_id"] in train_ids]
        test = [row for row in rows if row["account_id"] in test_ids]
        if not train or not test:
            raise ValueError("Split references accounts absent from feature table.")
        y_train = np.array([r["class_label"] == "coordinated_synthetic" for r in train], dtype=int)
        y_test = np.array([r["class_label"] == "coordinated_synthetic" for r in test], dtype=int)
        if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
            raise ValueError("Every train and test fold must contain both classes.")
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        model.fit(_float_matrix(train, FEATURE_COLUMNS), y_train)
        probability = model.predict_proba(_float_matrix(test, FEATURE_COLUMNS))[:, 1]
        metrics = _classification_metrics(y_test, probability)
        metrics["prevalence_scenarios"] = _prevalence_scenarios(metrics["sensitivity"], metrics["specificity"])
        result["splits"].append({"held_out_value": split["held_out_value"], **metrics})
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate account-level CCSF condition holdouts.")
    parser.add_argument("feature_csv")
    parser.add_argument("split_manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.feature_csv, args.split_manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
