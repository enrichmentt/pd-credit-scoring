"""Тесты первичной очистки и разбиения данных."""

import pandas as pd

from src import config
from src.data.prepare import clean_data, split_data


def test_clean_data_renames_target(raw_df):
    cleaned = clean_data(raw_df)
    assert config.TARGET in cleaned.columns
    assert config.RAW_TARGET not in cleaned.columns


def test_clean_data_drops_id(raw_df):
    cleaned = clean_data(raw_df)
    assert "ID" not in cleaned.columns


def test_clean_data_fixes_education_and_marriage():
    df = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "LIMIT_BAL": [10000, 20000, 30000, 40000],
            "SEX": [1, 2, 1, 2],
            "EDUCATION": [0, 5, 6, 2],
            "MARRIAGE": [0, 1, 2, 0],
            "AGE": [30, 40, 50, 60],
            **{c: [0, 0, 0, 0] for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]},
            **{f"BILL_AMT{i}": [0, 0, 0, 0] for i in range(1, 7)},
            **{f"PAY_AMT{i}": [0, 0, 0, 0] for i in range(1, 7)},
            config.RAW_TARGET: [0, 1, 0, 1],
        }
    )
    cleaned = clean_data(df)
    assert set(cleaned["EDUCATION"].unique()) == {2, 4}
    assert set(cleaned["MARRIAGE"].unique()) == {1, 2, 3}


def test_split_data_stratified(raw_df):
    cleaned = clean_data(raw_df)
    train_df, test_df = split_data(cleaned, test_size=0.25, random_state=0)
    assert len(train_df) + len(test_df) == len(cleaned)
    # Стратификация: доля дефолтов в выборках близка
    share_train = train_df[config.TARGET].mean()
    share_test = test_df[config.TARGET].mean()
    assert abs(share_train - share_test) < 0.05
