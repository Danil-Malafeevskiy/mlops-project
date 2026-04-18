"""Серия экспериментов: RandomForest vs LogisticRegression с логированием в MLflow."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_NAME = "fraud_rf_vs_lr"


def ensure_experiment(tracking_uri: str, name: str) -> None:
    """Проверка: старый Docker задал file:/mlflow/... — клиент пытался писать в /mlflow на хосте."""
    if not tracking_uri.startswith("http"):
        return
    client = MlflowClient(tracking_uri)
    exp = client.get_experiment_by_name(name)
    if exp is None:
        client.create_experiment(name)
        return
    loc = exp.artifact_location or ""
    if "mlflow-artifacts" in loc or loc.startswith("http"):
        return
    # Путь артефактов из контейнера /mlflow/... ошибочно воспринимается клиентом как локальный /mlflow
    parsed = urlparse(loc)
    if loc.startswith("file:") and parsed.path.startswith("/mlflow"):
        print(
            "Ошибка: эксперимент был создан старым сервером с artifact_location на file:/mlflow/....\n"
            "Исправление: пересоберите MLflow и сбросьте том SQLite, затем запустите снова:\n"
            "  docker compose -f docker/docker-compose.yml down -v\n"
            "  docker compose -f docker/docker-compose.yml up -d --build\n"
            "Либо удалите эксперимент «"
            + name
            + "» в UI MLflow и перезапустите скрипт.",
            file=sys.stderr,
        )
        sys.exit(1)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
    test = pd.read_parquet(ROOT / "data" / "processed" / "test.parquet")
    return train, test


def feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])
    return X, y


def metrics_dict(y_true, y_pred, y_proba_pos) -> dict[str, float]:
    """Метрики для класса Fraud (1) — в духе README проекта."""
    return {
        "recall_fraud": recall_score(y_true, y_pred, pos_label=1),
        "precision_fraud": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_fraud": f1_score(y_true, y_pred, pos_label=1),
        "pr_auc": average_precision_score(y_true, y_proba_pos),
        "roc_auc": roc_auc_score(y_true, y_proba_pos),
    }


def make_lr_pipeline(C: float, max_iter: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=max_iter,
                    random_state=42,
                    solver="liblinear",
                    dual=False,
                ),
            ),
        ]
    )


def make_rf_pipeline(n_estimators: int, max_depth: int | None) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def log_run(
    run_name: str,
    model: Pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    params: dict,
) -> dict[str, float]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    m = metrics_dict(y_test, y_pred, y_proba)
    example = X_train.head(5).astype("float64", copy=False)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(m)
        mlflow.set_tag("model_family", params.get("model_family", "unknown"))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=example,
        )

    return m


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    ensure_experiment(tracking_uri, EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train, test = load_splits()
    X_train, y_train = feature_target(train)
    X_test, y_test = feature_target(test)

    experiments: list[tuple[str, Pipeline, dict]] = [
        (
            "lr_C1",
            make_lr_pipeline(C=1.0, max_iter=2000),
            {"model_family": "logistic_regression", "C": 1.0, "max_iter": 2000},
        ),
        (
            "lr_C0.1",
            make_lr_pipeline(C=0.1, max_iter=2000),
            {"model_family": "logistic_regression", "C": 0.1, "max_iter": 2000},
        ),
        (
            "rf_trees100_depth12",
            make_rf_pipeline(n_estimators=100, max_depth=12),
            {
                "model_family": "random_forest",
                "n_estimators": 100,
                "max_depth": 12,
            },
        ),
        (
            "rf_trees150_depth15",
            make_rf_pipeline(n_estimators=150, max_depth=15),
            {
                "model_family": "random_forest",
                "n_estimators": 150,
                "max_depth": 15,
            },
        ),
    ]

    rows: list[dict] = []
    for run_name, pipe, params in experiments:
        print(f"Run: {run_name} ...", flush=True)
        m = log_run(run_name, pipe, X_train, y_train, X_test, y_test, params)
        rows.append({"run": run_name, **params, **m})
        print(f"  recall_fraud={m['recall_fraud']:.4f} pr_auc={m['pr_auc']:.4f}", flush=True)

    summary = pd.DataFrame(rows).sort_values("recall_fraud", ascending=False)
    print("\n=== Сводка (сортировка по recall_fraud) ===")
    print(summary.to_string(index=False))
    best = summary.iloc[0]
    print(f"\nЛучший прогон по recall_fraud: {best['run']} (для выбора в UI сравните также PR-AUC и F1).")


if __name__ == "__main__":
    main()
