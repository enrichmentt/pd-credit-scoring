"""Тесты feature engineering."""

import numpy as np
import pandas as pd

from src.features.build_features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    PAY_STATUS_COLS,
    build_features,
)
from src.data.prepare import clean_data


def test_build_features_adds_columns(raw_df):
    featured = build_features(clean_data(raw_df))
    for col in [
        "pay_delay_mean",
        "pay_delay_max",
        "months_delayed",
        "utilization_mean",
        "pay_ratio_mean",
        "bill_trend",
        "limit_per_age",
        "age_group",
    ]:
        assert col in featured.columns


def test_months_delayed_counts_positive_pay_status():
    row = {c: 0 for c in PAY_STATUS_COLS}
    row.update({"PAY_0": 2, "PAY_3": 1, "PAY_5": -1})
    df = pd.DataFrame(
        [
            {
                **row,
                "LIMIT_BAL": 100000,
                "AGE": 40,
                **{f"BILL_AMT{i}": 1000 for i in range(1, 7)},
                **{f"PAY_AMT{i}": 100 for i in range(1, 7)},
            }
        ]
    )
    featured = build_features(df)
    assert featured.loc[0, "months_delayed"] == 2
    assert featured.loc[0, "pay_delay_max"] == 2


def test_utilization_and_pay_ratio_bounds(raw_df):
    featured = build_features(clean_data(raw_df))
    assert (featured["utilization_mean"] >= 0).all()
    assert featured["pay_ratio_mean"].between(0, 1).all()


def test_age_group_bins():
    df = pd.DataFrame(
        {
            "LIMIT_BAL": [10000, 10000, 10000],
            "AGE": [20, 40, 70],
            **{c: [0, 0, 0] for c in PAY_STATUS_COLS},
            **{f"BILL_AMT{i}": [0, 0, 0] for i in range(1, 7)},
            **{f"PAY_AMT{i}": [0, 0, 0] for i in range(1, 7)},
        }
    )
    featured = build_features(df)
    assert list(featured["age_group"]) == ["<25", "35-45", "55+"]


def test_model_features_present(raw_df):
    featured = build_features(clean_data(raw_df))
    missing = [c for c in MODEL_FEATURES if c not in featured.columns]
    assert not missing, f"Нет признаков: {missing}"
    assert not np.isinf(featured[NUMERIC_FEATURES].to_numpy(dtype=float)).any()
    assert len(CATEGORICAL_FEATURES) > 0
