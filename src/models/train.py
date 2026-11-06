"""Обучение PD-модели: sklearn Pipeline + GridSearchCV + MLflow Tracking.

Запуск одного эксперимента:
    python -m src.models.train --model gradient_boosting --run-name gb_baseline
"""

import argparse
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.features.build_features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES

MODEL_REGISTRY = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
        {"clf__C": [0.1, 1.0, 10.0]},
    ),
    "logistic_regression_balanced": (
        LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        {"clf__C": [0.1, 1.0, 10.0]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
        {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=config.RANDOM_STATE),
        {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1]},
    ),
    "gradient_boosting_deep": (
        GradientBoostingClassifier(random_state=config.RANDOM_STATE, subsample=0.8),
        {"clf__n_estimators": [150], "clf__max_depth": [3, 5]},
    ),
}


def build_pipeline(model_type: str) -> Pipeline:
    """Sklearn Pipeline: Imputer + Scaler / OneHotEncoder + классификатор."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Неизвестная модель: {model_type}. Доступны: {list(MODEL_REGISTRY)}")
    estimator, _ = MODEL_REGISTRY[model_type]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("clf", estimator)])


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Ключевые метрики бинарной классификации."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def plot_roc_curve(y_true, y_proba, save_path) -> None:
    """Строит ROC-кривую и сохраняет в PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", label="Случайный классификатор")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривая PD-модели")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def load_processed():
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    return train_df, test_df


def train_experiment(
    model_type: str,
    run_name: str | None = None,
    param_grid: dict | None = None,
    cv: int = 3,
    save_artifacts: bool = False,
) -> dict:
    """Один эксперимент: GridSearchCV + оценка на тесте + логирование в MLflow."""
    train_df, test_df = load_processed()
    X_train, y_train = train_df[MODEL_FEATURES], train_df[config.TARGET]
    X_test, y_test = test_df[MODEL_FEATURES], test_df[config.TARGET]

    pipeline = build_pipeline(model_type)
    _, default_grid = MODEL_REGISTRY[model_type]
    grid = param_grid or default_grid

    search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    best = search.best_estimator_

    y_proba = best.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    # --- MLflow ---
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    run_name = run_name or model_type

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    roc_path = config.FIGURES_DIR / f"roc_{run_name}.png"
    plot_roc_curve(y_test, y_proba, roc_path)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("cv", cv)
        mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
        mlflow.log_metric("best_cv_roc_auc", float(search.best_score_))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(roc_path))
        log_kwargs = {
            "input_example": X_train.head(3),
            "serialization_format": "cloudpickle",
        }
        try:
            mlflow.sklearn.log_model(best, name="model", **log_kwargs)
        except TypeError:  # mlflow < 3.x
            mlflow.sklearn.log_model(best, artifact_path="model")

    # Артефакты лучшей модели для DVC/API
    if save_artifacts:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best, config.MODEL_PATH)
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"model_type": model_type, "best_params": search.best_params_, **metrics},
                f,
                indent=2,
                ensure_ascii=False,
            )
        # Базовое распределение скоров на train — для мониторинга дрифта
        train_proba = best.predict_proba(X_train)[:, 1]
        pd.DataFrame({"train_proba": train_proba}).to_csv(config.TRAIN_SCORES_PATH, index=False)

    result = {"run_name": run_name, "best_params": search.best_params_, **metrics}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description="Обучение PD-модели")
    parser.add_argument("--model", default=None, choices=list(MODEL_REGISTRY))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--cv", type=int, default=None)
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Минимальная сетка (для dvc repro)")
    args = parser.parse_args()

    params = {}
    params_file = config.PROJECT_ROOT / "params.yaml"
    if params_file.exists():
        params = yaml.safe_load(params_file.read_text(encoding="utf-8")).get("train", {})

    model_type = args.model or params.get("model", "gradient_boosting")
    cv = args.cv if args.cv is not None else params.get("cv", 3)
    run_name = args.run_name or params.get("run_name", model_type)

    param_grid = None
    if args.quick:
        param_grid = {k: v[:1] for k, v in MODEL_REGISTRY[model_type][1].items()}

    train_experiment(
        model_type,
        run_name=run_name,
        param_grid=param_grid,
        cv=cv,
        save_artifacts=args.save_artifacts or bool(params),
    )


if __name__ == "__main__":
    main()
