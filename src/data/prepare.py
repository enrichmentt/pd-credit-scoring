"""Загрузка, первичная очистка и разбиение данных на train/test.

Запуск: python -m src.data.prepare
"""

import argparse

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src import config
from src.features.build_features import build_features
from src.validation.schema import validate_raw


def load_data(path=None) -> pd.DataFrame:
    """Читает сырой CSV с данными UCI Credit Card."""
    path = path or config.DATA_RAW
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Первичная очистка датасета UCI Credit Card.

    - удаляет техническую колонку ID;
    - приводит редкие/недокументированные категории EDUCATION (0, 5, 6) к 4 («другое»);
    - приводит MARRIAGE = 0 к 3 («другое»);
    - переименовывает целевую переменную в ``default``.
    """
    df = df.copy()
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    df = df.rename(columns={config.RAW_TARGET: config.TARGET})
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = config.RANDOM_STATE):
    """Стратифицированное разбиение на train/test по целевой переменной."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[config.TARGET]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare(raw_path=None, test_size: float = 0.2, random_state: int = config.RANDOM_STATE):
    """Полный цикл подготовки: сырые данные -> очищенные признаки -> train/test."""
    raw = load_data(raw_path)
    validate_raw(raw)  # валидация входящих данных до обработки

    cleaned = clean_data(raw)
    featured = build_features(cleaned)

    train_df, test_df = split_data(featured, test_size=test_size, random_state=random_state)

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.TRAIN_PATH, index=False)
    test_df.to_csv(config.TEST_PATH, index=False)
    print(
        f"Готово: train {train_df.shape} -> {config.TRAIN_PATH}, "
        f"test {test_df.shape} -> {config.TEST_PATH}"
    )
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Подготовка данных для PD-модели")
    parser.add_argument("--raw-path", default=None, help="Путь к сырому CSV")
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    args = parser.parse_args()

    params = {}
    params_file = config.PROJECT_ROOT / "params.yaml"
    if params_file.exists():
        params = yaml.safe_load(params_file.read_text(encoding="utf-8")).get("prepare", {})

    test_size = args.test_size if args.test_size is not None else params.get("test_size", 0.2)
    random_state = (
        args.random_state
        if args.random_state is not None
        else params.get("random_state", config.RANDOM_STATE)
    )

    prepare(raw_path=args.raw_path, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    main()
