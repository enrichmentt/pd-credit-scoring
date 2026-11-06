"""Инференс: загрузка обученного пайплайна и скоринг новых данных."""

import joblib
import pandas as pd

from src import config
from src.features.build_features import MODEL_FEATURES, build_features


def load_model(path=None):
    """Загружает сохранённый sklearn Pipeline."""
    return joblib.load(path or config.MODEL_PATH)


def predict_proba(model, raw_features: pd.DataFrame) -> pd.Series:
    """Вероятность дефолта для датафрейма сырых признаков (без ID/таргета)."""
    featured = build_features(raw_features)
    proba = model.predict_proba(featured[MODEL_FEATURES])[:, 1]
    return pd.Series(proba, index=raw_features.index)
