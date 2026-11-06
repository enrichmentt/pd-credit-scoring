"""Feature engineering для PD-модели.

Из сырых колонок датасета строятся агрегированные признаки по истории
платежей, утилизации лимита и биннинг возраста.
"""

import numpy as np
import pandas as pd

PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

EPS = 1.0  # защита от деления на ноль


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет производные признаки к очищенному датасету.

    Новые признаки:
    - pay_delay_mean / pay_delay_max: средняя и максимальная просрочка за 6 мес.;
    - months_delayed: число месяцев с просрочкой (PAY_x > 0);
    - utilization_mean: средняя утилизация кредитного лимита;
    - pay_ratio_mean: средняя доля погашения счёта;
    - bill_trend: динамика долга (последний счёт минус первый);
    - age_group: биннинг возраста;
    - limit_per_age: лимит на год возраста.
    """
    df = df.copy()

    # История просрочек
    df["pay_delay_mean"] = df[PAY_STATUS_COLS].mean(axis=1)
    df["pay_delay_max"] = df[PAY_STATUS_COLS].max(axis=1)
    df["months_delayed"] = (df[PAY_STATUS_COLS] > 0).sum(axis=1)

    # Утилизация лимита и дисциплина платежей
    bills = df[BILL_COLS].clip(lower=0)
    df["utilization_mean"] = bills.div(df["LIMIT_BAL"] + EPS, axis=0).mean(axis=1)
    pay_ratio = df[PAY_AMT_COLS].to_numpy() / (bills.to_numpy() + EPS)
    df["pay_ratio_mean"] = np.clip(pay_ratio, 0, 1).mean(axis=1)
    df["bill_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]

    # Демография
    df["limit_per_age"] = df["LIMIT_BAL"] / (df["AGE"] + EPS)
    df["age_group"] = pd.cut(
        df["AGE"],
        bins=[0, 25, 35, 45, 55, np.inf],
        labels=["<25", "25-35", "35-45", "45-55", "55+"],
    ).astype(str)

    return df


# Признаки, подаваемые в модель
NUMERIC_FEATURES = (
    [
        "LIMIT_BAL",
        "AGE",
        "pay_delay_mean",
        "pay_delay_max",
        "months_delayed",
        "utilization_mean",
        "pay_ratio_mean",
        "bill_trend",
        "limit_per_age",
    ]
    + PAY_STATUS_COLS
    + BILL_COLS
    + PAY_AMT_COLS
)

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE", "age_group"]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
