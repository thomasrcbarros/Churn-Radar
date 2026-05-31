"""Ingestão de dados do dataset Telecom Churn via kagglehub.

O acesso ao Kaggle pode estar bloqueado em ambientes de nuvem; nesse caso o
download deve ser executado localmente. As credenciais são lidas de um arquivo
.env (nunca versionado).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from . import config


def _load_credentials() -> None:
    """Carrega KAGGLE_USERNAME/KAGGLE_KEY do .env para o ambiente."""
    load_dotenv(config.ROOT_DIR / ".env")
    if not os.environ.get("KAGGLE_KEY") or not os.environ.get("KAGGLE_USERNAME"):
        print(
            "[aviso] KAGGLE_USERNAME/KAGGLE_KEY não encontrados no ambiente. "
            "Crie um arquivo .env a partir de .env.example antes de baixar os dados."
        )


def download_dataset() -> Path:
    """Baixa o dataset via kagglehub e retorna o diretório local.

    Usa kagglehub==1.0.0 (ver requirements.txt) para compatibilidade.
    """
    _load_credentials()
    try:
        import kagglehub

        path = kagglehub.dataset_download(config.KAGGLE_DATASET)
        return Path(path)
    except Exception as exc:  # rede bloqueada na nuvem, credenciais, etc.
        raise RuntimeError(
            "Falha ao baixar o dataset via kagglehub. O acesso ao Kaggle costuma "
            "estar bloqueado em ambientes de nuvem — rode este passo localmente "
            "com um .env válido. Detalhe: " + str(exc)
        ) from exc


def _find_csv(directory: Path) -> Path:
    csvs = list(directory.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {directory}")
    return csvs[0]


def load_data() -> pd.DataFrame:
    """Baixa (se necessário) e carrega o dataset de churn como DataFrame."""
    directory = download_dataset()
    csv_path = _find_csv(directory)
    df = pd.read_csv(csv_path)
    return df


if __name__ == "__main__":
    frame = load_data()
    print(frame.shape)
    print(frame.head())
