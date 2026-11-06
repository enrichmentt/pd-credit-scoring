"""Тесты валидации данных Pandera.

test_schema_rejects_anomalies — тест, который проваливается, если валидация
НЕ ловит аномалии в новых данных (null, значения вне диапазона, битые категории).
"""

import pytest
from pandera.errors import SchemaErrors

from src.validation.schema import validate_raw


def test_schema_accepts_valid_data(raw_df):
    validated = validate_raw(raw_df)
    assert len(validated) == len(raw_df)


def test_schema_rejects_nulls(raw_df):
    broken = raw_df.copy()
    broken.loc[0, "LIMIT_BAL"] = None
    with pytest.raises(SchemaErrors):
        validate_raw(broken)


def test_schema_rejects_out_of_range_age(raw_df):
    broken = raw_df.copy()
    broken.loc[0, "AGE"] = 5  # слишком юный заёмщик
    broken.loc[1, "AGE"] = 150  # и слишком возрастной
    with pytest.raises(SchemaErrors):
        validate_raw(broken)


def test_schema_rejects_invalid_categories(raw_df):
    broken = raw_df.copy()
    broken.loc[0, "SEX"] = 7
    with pytest.raises(SchemaErrors):
        validate_raw(broken)


def test_schema_rejects_invalid_target(raw_df):
    broken = raw_df.copy()
    broken.loc[0, "default.payment.next.month"] = 2
    with pytest.raises(SchemaErrors):
        validate_raw(broken)


def test_schema_rejects_anomalies(raw_df):
    """Сводный тест: валидация обязана отклонить данные с аномалиями.

    Если этот тест падает — значит, правила валидации перестали работать,
    и битые новые данные могут незаметно уйти в обучение.
    """
    anomalies = raw_df.copy()
    anomalies.loc[0, "LIMIT_BAL"] = None  # пропуск
    anomalies.loc[1, "AGE"] = 999  # вне диапазона
    anomalies.loc[2, "EDUCATION"] = 42  # неизвестная категория
    with pytest.raises(SchemaErrors):
        validate_raw(anomalies)
