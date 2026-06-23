"""Carregamento dos CSVs canonicos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ARQUIVOS_CANONICOS = {
    "participantes": "participantes.csv",
    "jogos": "jogos.csv",
    "palpites": "palpites.csv",
    "resultados": "resultados.csv",
}

COLUNAS_OBRIGATORIAS = {
    "participantes": {"participante_id", "rotulo"},
    "jogos": {"jogo_id", "rodada"},
    "palpites": {
        "participante_id",
        "jogo_id",
        "gols_a_palpite",
        "gols_b_palpite",
    },
    "resultados": {"jogo_id", "gols_a_real", "gols_b_real", "status"},
}


def _validar_colunas(nome: str, df: pd.DataFrame) -> None:
    colunas_ausentes = COLUNAS_OBRIGATORIAS[nome].difference(df.columns)
    if colunas_ausentes:
        colunas = ", ".join(sorted(colunas_ausentes))
        raise ValueError(f"{ARQUIVOS_CANONICOS[nome]} sem colunas obrigatorias: {colunas}.")


def _converter_colunas_inteiras(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    for coluna in colunas:
        df[coluna] = pd.to_numeric(df[coluna], errors="raise").astype(int)
    return df


def carregar_dados(data_dir: str) -> dict:
    """
    Carrega os quatro CSVs canonicos do projeto.

    Args:
        data_dir (str): Diretorio contendo os CSVs canonicos.

    Returns:
        dict: DataFrames de participantes, jogos, palpites e resultados.
    """
    base = Path(data_dir)
    dados = {}

    for nome, arquivo in ARQUIVOS_CANONICOS.items():
        caminho = base / arquivo
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo obrigatorio nao encontrado: {caminho}")
        dados[nome] = pd.read_csv(caminho, dtype={"participante_id": str, "jogo_id": str})
        _validar_colunas(nome, dados[nome])

    dados["palpites"] = _converter_colunas_inteiras(
        dados["palpites"],
        ["gols_a_palpite", "gols_b_palpite"],
    )
    dados["resultados"] = _converter_colunas_inteiras(
        dados["resultados"],
        ["gols_a_real", "gols_b_real"],
    )

    return dados
