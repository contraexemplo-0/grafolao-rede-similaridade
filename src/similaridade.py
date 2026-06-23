"""Calculo de similaridade entre palpites e participantes."""

from __future__ import annotations

from itertools import combinations
from numbers import Integral, Real
from typing import Sequence

import pandas as pd

from src.config import C_NORMALIZACAO_PLACAR


COLUNAS_PALPITES = {
    "participante_id",
    "jogo_id",
    "gols_a_palpite",
    "gols_b_palpite",
}

COLUNAS_PARES = [
    "participante_u",
    "participante_v",
    "jogos_comparaveis",
    "total_jogos",
    "sim_media",
    "cobertura",
    "sim_final",
]


def _validar_c(c: int) -> None:
    if not isinstance(c, Integral) or isinstance(c, bool) or c <= 0:
        raise ValueError("A constante de normalizacao c deve ser um inteiro positivo.")


def _validar_gols(gols: int, nome: str) -> int:
    if not isinstance(gols, Integral) or isinstance(gols, bool):
        raise ValueError(f"{nome} deve ser um inteiro nao negativo.")
    if gols < 0:
        raise ValueError(f"{nome} nao pode ser negativo.")
    return int(gols)


def _validar_palpite(palpite: tuple[int, int] | list[int]) -> tuple[int, int]:
    if not isinstance(palpite, (tuple, list)) or len(palpite) != 2:
        raise ValueError("Palpite deve ser uma tupla ou lista com dois valores.")

    gols_a = _validar_gols(palpite[0], "gols_a")
    gols_b = _validar_gols(palpite[1], "gols_b")
    return gols_a, gols_b


def _validar_colunas_palpites(palpites_df: pd.DataFrame) -> None:
    colunas_ausentes = COLUNAS_PALPITES.difference(palpites_df.columns)
    if colunas_ausentes:
        colunas = ", ".join(sorted(colunas_ausentes))
        raise ValueError(f"DataFrame de palpites sem colunas obrigatorias: {colunas}.")


def _validar_jogos_considerados(jogos_considerados: Sequence[str]) -> list[str]:
    jogos = list(jogos_considerados)
    if not jogos:
        raise ValueError("jogos_considerados nao pode estar vazio.")
    return jogos


def _validar_cobertura_gamma(cobertura_gamma: float) -> float:
    if (
        not isinstance(cobertura_gamma, Real)
        or isinstance(cobertura_gamma, bool)
        or cobertura_gamma <= 0
    ):
        raise ValueError("cobertura_gamma deve ser um numero positivo.")
    return float(cobertura_gamma)


def resultado_previsto(gols_a: int, gols_b: int) -> str:
    """
    Identifica o resultado previsto de um palpite.

    Args:
        gols_a (int): Gols previstos para o time A.
        gols_b (int): Gols previstos para o time B.

    Returns:
        str: "A" para vitoria do time A, "E" para empate ou "B" para vitoria do time B.
    """
    gols_a_validado = _validar_gols(gols_a, "gols_a")
    gols_b_validado = _validar_gols(gols_b, "gols_b")

    if gols_a_validado > gols_b_validado:
        return "A"
    if gols_a_validado == gols_b_validado:
        return "E"
    return "B"


def proximidade_placar(
    palpite_p: tuple[int, int],
    palpite_q: tuple[int, int],
    c: int = C_NORMALIZACAO_PLACAR,
) -> float:
    """
    Calcula a proximidade numerica entre dois placares previstos.

    Args:
        palpite_p (tuple[int, int]): Placar previsto pelo primeiro participante.
        palpite_q (tuple[int, int]): Placar previsto pelo segundo participante.
        c (int): Constante de normalizacao das diferencas de placar.

    Returns:
        float: Proximidade de placar no intervalo [0, 1].
    """
    _validar_c(c)
    gols_a_p, gols_b_p = _validar_palpite(palpite_p)
    gols_a_q, gols_b_q = _validar_palpite(palpite_q)

    distancia = abs(gols_a_p - gols_a_q) + abs(gols_b_p - gols_b_q)
    return 1 - min(distancia, c) / c


def proximidade_saldo(
    palpite_p: tuple[int, int],
    palpite_q: tuple[int, int],
    c: int = C_NORMALIZACAO_PLACAR,
) -> float:
    """
    Calcula a proximidade entre os saldos de dois placares previstos.

    Args:
        palpite_p (tuple[int, int]): Placar previsto pelo primeiro participante.
        palpite_q (tuple[int, int]): Placar previsto pelo segundo participante.
        c (int): Constante de normalizacao das diferencas de saldo.

    Returns:
        float: Proximidade de saldo no intervalo [0, 1].
    """
    _validar_c(c)
    gols_a_p, gols_b_p = _validar_palpite(palpite_p)
    gols_a_q, gols_b_q = _validar_palpite(palpite_q)

    saldo_p = gols_a_p - gols_b_p
    saldo_q = gols_a_q - gols_b_q
    distancia = abs(saldo_p - saldo_q)
    return 1 - min(distancia, c) / c


def similaridade_jogo(
    palpite_p: tuple[int, int],
    palpite_q: tuple[int, int],
    c: int = C_NORMALIZACAO_PLACAR,
) -> float:
    """
    Calcula a similaridade entre dois palpites para o mesmo jogo.

    Args:
        palpite_p (tuple[int, int]): Placar previsto pelo primeiro participante.
        palpite_q (tuple[int, int]): Placar previsto pelo segundo participante.
        c (int): Constante de normalizacao das diferencas de placar.

    Returns:
        float: Valor de similaridade no intervalo [0, 1].
    """
    _validar_c(c)
    gols_a_p, gols_b_p = _validar_palpite(palpite_p)
    gols_a_q, gols_b_q = _validar_palpite(palpite_q)

    prox_placar = proximidade_placar((gols_a_p, gols_b_p), (gols_a_q, gols_b_q), c)

    if resultado_previsto(gols_a_p, gols_b_p) != resultado_previsto(gols_a_q, gols_b_q):
        return 0.25 * prox_placar

    prox_saldo = proximidade_saldo((gols_a_p, gols_b_p), (gols_a_q, gols_b_q), c)
    return 0.50 + 0.25 * prox_saldo + 0.25 * prox_placar


def similaridade_participantes(
    participante_u: str,
    participante_v: str,
    palpites_df,
    jogos_considerados: list[str],
    c: int = C_NORMALIZACAO_PLACAR,
    cobertura_gamma: float = 1.0,
) -> dict:
    """
    Calcula a similaridade ajustada por cobertura entre dois participantes.

    Args:
        participante_u (str): ID do primeiro participante.
        participante_v (str): ID do segundo participante.
        palpites_df: DataFrame com palpites canonicos.
        jogos_considerados (list[str]): Jogos incluidos no recorte de analise.
        c (int): Constante de normalizacao das diferencas de placar.
        cobertura_gamma (float): Expoente aplicado a cobertura. O padrao 1.0
            preserva a penalizacao linear oficial.

    Returns:
        dict: Similaridade media, cobertura e similaridade final do par.
    """
    _validar_c(c)
    gamma = _validar_cobertura_gamma(cobertura_gamma)
    jogos = _validar_jogos_considerados(jogos_considerados)
    _validar_colunas_palpites(palpites_df)

    palpites = palpites_df[
        (palpites_df["participante_id"].isin([participante_u, participante_v]))
        & (palpites_df["jogo_id"].isin(jogos))
    ]

    palpites_por_chave = {
        (linha.participante_id, linha.jogo_id): (
            linha.gols_a_palpite,
            linha.gols_b_palpite,
        )
        for linha in palpites.itertuples(index=False)
    }

    similaridades = []
    for jogo_id in jogos:
        palpite_u = palpites_por_chave.get((participante_u, jogo_id))
        palpite_v = palpites_por_chave.get((participante_v, jogo_id))

        if palpite_u is None or palpite_v is None:
            continue

        similaridades.append(similaridade_jogo(palpite_u, palpite_v, c))

    jogos_comparaveis = len(similaridades)
    total_jogos = len(jogos)

    if jogos_comparaveis == 0:
        sim_media = 0.0
        cobertura = 0.0
        sim_final = 0.0
    else:
        sim_media = sum(similaridades) / jogos_comparaveis
        cobertura = jogos_comparaveis / total_jogos
        sim_final = sim_media * (cobertura**gamma)

    return {
        "participante_u": participante_u,
        "participante_v": participante_v,
        "jogos_comparaveis": jogos_comparaveis,
        "total_jogos": total_jogos,
        "sim_media": sim_media,
        "cobertura": cobertura,
        "sim_final": sim_final,
    }


def calcular_pares_similaridade(
    participantes: list[str],
    palpites_df,
    jogos_considerados: list[str],
    c: int = C_NORMALIZACAO_PLACAR,
    cobertura_gamma: float = 1.0,
):
    """
    Calcula similaridade para todos os pares unicos de participantes.

    Args:
        participantes (list[str]): IDs dos participantes, na ordem desejada.
        palpites_df: DataFrame com palpites canonicos.
        jogos_considerados (list[str]): Jogos incluidos no recorte de analise.
        c (int): Constante de normalizacao das diferencas de placar.
        cobertura_gamma (float): Expoente aplicado a cobertura. O padrao 1.0
            preserva a penalizacao linear oficial.

    Returns:
        pandas.DataFrame: Tabela com uma linha por par unico de participantes.
    """
    linhas = [
        similaridade_participantes(
            participante_u,
            participante_v,
            palpites_df,
            jogos_considerados,
            c,
            cobertura_gamma=cobertura_gamma,
        )
        for participante_u, participante_v in combinations(participantes, 2)
    ]

    return pd.DataFrame(linhas, columns=COLUNAS_PARES)


def calcular_matriz_similaridade(
    participantes: list[str],
    pares_df,
):
    """
    Monta a matriz quadrada e simetrica de similaridade entre participantes.

    Args:
        participantes (list[str]): IDs dos participantes, na ordem desejada.
        pares_df: DataFrame com os pares e a coluna `sim_final`.

    Returns:
        pandas.DataFrame: Matriz de similaridade indexada por participante.
    """
    colunas_ausentes = set(COLUNAS_PARES).difference(pares_df.columns)
    if colunas_ausentes:
        colunas = ", ".join(sorted(colunas_ausentes))
        raise ValueError(f"DataFrame de pares sem colunas obrigatorias: {colunas}.")

    matriz = pd.DataFrame(0.0, index=participantes, columns=participantes)

    for participante in participantes:
        matriz.loc[participante, participante] = 1.0

    for linha in pares_df.itertuples(index=False):
        sim_final = float(linha.sim_final)
        matriz.loc[linha.participante_u, linha.participante_v] = sim_final
        matriz.loc[linha.participante_v, linha.participante_u] = sim_final

    return matriz
