"""Серия экспериментов с разными алгоритмами и гиперпараметрами (MLflow).

Запуск: python -m src.models.run_experiments
Результаты смотреть: mlflow ui --backend-store-uri ./mlruns
"""

from src.models.train import train_experiment

EXPERIMENTS = [
    {"model_type": "logistic_regression", "run_name": "logreg_baseline"},
    {"model_type": "logistic_regression_balanced", "run_name": "logreg_balanced"},
    {"model_type": "random_forest", "run_name": "rf_grid"},
    {"model_type": "gradient_boosting", "run_name": "gb_grid"},
    {"model_type": "gradient_boosting_deep", "run_name": "gb_deep_subsample"},
]


def main():
    results = []
    for exp in EXPERIMENTS:
        print(f"\n=== Эксперимент: {exp['run_name']} ===")
        results.append(train_experiment(**exp))

    print("\n=== Сводка по экспериментам ===")
    for r in sorted(results, key=lambda x: -x["roc_auc"]):
        print(
            f"{r['run_name']:>25}: ROC-AUC={r['roc_auc']:.4f} "
            f"F1={r['f1']:.4f} Precision={r['precision']:.4f} Recall={r['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
