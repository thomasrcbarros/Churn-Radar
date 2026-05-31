"""Preparação de features: separação X/y e split treino/teste estratificado."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def split_xy(df: pd.DataFrame):
    """Separa features (X) e alvo (y)."""
    y = df[config.TARGET_COL]
    X = df.drop(columns=[config.TARGET_COL])
    return X, y


def make_splits(df: pd.DataFrame):
    """Retorna X_train, X_test, y_train, y_test com split estratificado."""
    X, y = split_xy(df)
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
