"""Схема валидации входящих данных (Pandera).

Проверяет сырой датасет UCI Credit Card: наличие колонок, типы,
отсутствие пропусков и допустимые диапазоны значений.
"""

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

RAW_SCHEMA = DataFrameSchema(
    columns={
        "ID": Column(int, Check.greater_than(0), nullable=False),
        "LIMIT_BAL": Column(float, Check.in_range(0, 10_000_000), nullable=False, coerce=True),
        "SEX": Column(int, Check.isin([1, 2]), nullable=False, coerce=True),
        "EDUCATION": Column(int, Check.isin([0, 1, 2, 3, 4, 5, 6]), nullable=False, coerce=True),
        "MARRIAGE": Column(int, Check.isin([0, 1, 2, 3]), nullable=False, coerce=True),
        "AGE": Column(int, Check.in_range(18, 100), nullable=False, coerce=True),
        "default.payment.next.month": Column(int, Check.isin([0, 1]), nullable=False, coerce=True),
        **{
            col: Column(int, Check.in_range(-2, 9), nullable=False, coerce=True)
            for col in PAY_STATUS_COLS
        },
        **{col: Column(float, nullable=False, coerce=True) for col in BILL_COLS},
        **{
            col: Column(float, Check.greater_than_or_equal_to(0), nullable=False, coerce=True)
            for col in PAY_AMT_COLS
        },
    },
    strict=False,
    name="uci_credit_card_raw",
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Валидирует сырой датафрейм. Бросает pandera.errors.SchemaError при аномалиях."""
    return RAW_SCHEMA.validate(df, lazy=True)
