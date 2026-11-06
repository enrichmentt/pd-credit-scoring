"""Мониторинг дрифта: Population Stability Index (PSI).

Скрипт имитирует поступление новых данных (часть тестовой выборки),
прогоняет их через модель (локально или через API) и считает PSI:
- по распределению вероятностей дефолта (скоры train vs «новые»);
- по ключевым признакам.

Интерпретация PSI: < 0.1 — дрифта нет; 0.1–0.25 — умеренный; > 0.25 — значительный.

Запуск локально:
    python -m src.monitoring.drift --sample-size 2000
Запуск через API (сервис должен быть запущен):
    python -m src.monitoring.drift --api-url http://localhost:8000/predict
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
import requests

from src import config
from src.models.predict import predict_proba

PSI_FEATURES = ["LIMIT_BAL", "AGE", "pay_delay_mean", "pay_delay_max", "utilization_mean"]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index между двумя распределениями.

    Бины строятся по квантилям референсного (expected) распределения.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    quantiles = np.unique(quantiles)

    exp_counts, _ = np.histogram(expected, bins=quantiles)
    act_counts, _ = np.histogram(actual, bins=quantiles)

    # Сглаживание, чтобы избежать деления на ноль
    exp_share = np.maximum(exp_counts / max(exp_counts.sum(), 1), 1e-6)
    act_share = np.maximum(act_counts / max(act_counts.sum(), 1), 1e-6)

    return float(np.sum((exp_share - act_share) * np.log(exp_share / act_share)))


def score_via_api(df_raw: pd.DataFrame, api_url: str) -> np.ndarray:
    """Отправляет записи на /predict и собирает вероятности дефолта."""
    proba = []
    payload_cols = [c for c in df_raw.columns if c not in ("ID", config.TARGET)]
    for record in df_raw[payload_cols].to_dict(orient="records"):
        resp = requests.post(api_url, json=record, timeout=30)
        resp.raise_for_status()
        proba.append(resp.json()["probability"])
    return np.array(proba)


def main() -> dict:
    parser = argparse.ArgumentParser(description="Мониторинг дрифта (PSI)")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument(
        "--api-url", default=None, help="URL /predict; без него — локальный скоринг"
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    # Референс: распределение скоров и признаков на train
    train_scores = pd.read_csv(config.TRAIN_SCORES_PATH)["train_proba"].to_numpy()
    train_df = pd.read_csv(config.TRAIN_PATH)

    # «Новые» данные: случайная часть тестовой выборки
    test_df = pd.read_csv(config.TEST_PATH)
    new_df = test_df.sample(n=min(args.sample_size, len(test_df)), random_state=args.seed)

    if args.api_url:
        raw_new = new_df.drop(columns=[config.TARGET])
        # Приводим обработанные признаки обратно к «сырому» виду для API
        raw_new = raw_new[
            [
                c
                for c in raw_new.columns
                if not c.startswith(
                    (
                        "pay_delay",
                        "months_",
                        "utilization",
                        "pay_ratio",
                        "bill_trend",
                        "limit_per_age",
                        "age_group",
                    )
                )
            ]
        ]
        new_scores = score_via_api(raw_new, args.api_url)
    else:
        model = joblib.load(config.MODEL_PATH)
        new_scores = predict_proba(model, new_df.drop(columns=[config.TARGET])).to_numpy()

    report = {"mode": "api" if args.api_url else "local", "sample_size": int(len(new_df))}

    # PSI по скорам
    score_psi = psi(train_scores, new_scores)
    report["score_psi"] = round(score_psi, 4)

    # PSI по ключевым признакам
    feature_psi = {}
    for col in PSI_FEATURES:
        if col in train_df.columns and col in new_df.columns:
            feature_psi[col] = round(psi(train_df[col].to_numpy(), new_df[col].to_numpy()), 4)
    report["feature_psi"] = feature_psi

    def verdict(v):
        return "OK" if v < 0.1 else ("MODERATE DRIFT" if v < 0.25 else "SIGNIFICANT DRIFT")

    report["verdict"] = {
        "scores": verdict(score_psi),
        **{f: verdict(v) for f, v in feature_psi.items()},
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
