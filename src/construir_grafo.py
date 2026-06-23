"""Construcao dos grafos de similaridade."""

from __future__ import annotations

from itertools import combinations
from numbers import Real

import networkx as nx
import pandas as pd

from src.config import THETA_SIMILARIDADE


COLUNAS_PARES_OBRIGATORIAS = {"participante_u", "participante_v", "sim_final"}
COLUNAS_JOGOS_OBRIGATORIAS = {"jogo_id", "rodada"}


def _validar_theta(theta: float) -> None:
    if not isinstance(theta, Real) or isinstance(theta, bool):
        raise ValueError("theta deve ser um numero no intervalo [0, 1].")
    if theta < 0 or theta > 1:
        raise ValueError("theta deve estar no intervalo [0, 1].")


def _validar_similaridade(valor: float, contexto: str) -> float:
    if not isinstance(valor, Real) or isinstance(valor, bool):
        raise ValueError(f"Similaridade invalida em {contexto}: deve ser numerica.")

    similaridade = float(valor)
    if similaridade < 0 or similaridade > 1:
        raise ValueError(f"Similaridade invalida em {contexto}: deve estar entre 0 e 1.")
    return similaridade


def _validar_matriz_similaridade(
    participantes: list[str],
    matriz_similaridade: pd.DataFrame,
) -> None:
    if not isinstance(matriz_similaridade, pd.DataFrame):
        raise ValueError("matriz_similaridade deve ser um pandas.DataFrame.")

    if matriz_similaridade.shape[0] != matriz_similaridade.shape[1]:
        raise ValueError("matriz_similaridade deve ser quadrada.")

    participantes_ausentes = [
        participante
        for participante in participantes
        if participante not in matriz_similaridade.index
        or participante not in matriz_similaridade.columns
    ]
    if participantes_ausentes:
        ausentes = ", ".join(participantes_ausentes)
        raise ValueError(f"matriz_similaridade nao contem participantes: {ausentes}.")

    for participante_u, participante_v in combinations(participantes, 2):
        sim_uv = _validar_similaridade(
            matriz_similaridade.loc[participante_u, participante_v],
            f"{participante_u}-{participante_v}",
        )
        sim_vu = _validar_similaridade(
            matriz_similaridade.loc[participante_v, participante_u],
            f"{participante_v}-{participante_u}",
        )
        if sim_uv != sim_vu:
            raise ValueError("matriz_similaridade deve ser simetrica.")

    for participante in participantes:
        _validar_similaridade(
            matriz_similaridade.loc[participante, participante],
            f"{participante}-{participante}",
        )


def construir_grafo_similaridade(
    participantes: list[str],
    matriz_similaridade,
    theta: float = THETA_SIMILARIDADE,
):
    """
    Constroi o grafo de similaridade a partir de uma matriz de similaridade.

    Args:
        participantes (list[str]): IDs dos participantes que serao vertices.
        matriz_similaridade: DataFrame quadrado com similaridades entre participantes.
        theta (float): Limiar minimo para criacao de arestas.

    Returns:
        networkx.Graph: Grafo nao direcionado, simples, ponderado e filtrado.
    """
    _validar_theta(theta)
    _validar_matriz_similaridade(participantes, matriz_similaridade)

    grafo = nx.Graph()
    grafo.add_nodes_from(participantes)

    for participante_u, participante_v in combinations(participantes, 2):
        similaridade = float(matriz_similaridade.loc[participante_u, participante_v])

        if similaridade >= theta:
            grafo.add_edge(
                participante_u,
                participante_v,
                weight=similaridade,
                distance=1 - similaridade,
            )

    return grafo


def construir_grafo_de_pares(
    participantes: list[str],
    pares_df,
    theta: float = THETA_SIMILARIDADE,
):
    """
    Constroi o grafo de similaridade a partir da tabela de pares.

    Args:
        participantes (list[str]): IDs dos participantes que serao vertices.
        pares_df: DataFrame contendo participante_u, participante_v e sim_final.
        theta (float): Limiar minimo para criacao de arestas.

    Returns:
        networkx.Graph: Grafo nao direcionado, simples, ponderado e filtrado.
    """
    _validar_theta(theta)
    if not isinstance(pares_df, pd.DataFrame):
        raise ValueError("pares_df deve ser um pandas.DataFrame.")

    colunas_ausentes = COLUNAS_PARES_OBRIGATORIAS.difference(pares_df.columns)
    if colunas_ausentes:
        colunas = ", ".join(sorted(colunas_ausentes))
        raise ValueError(f"pares_df sem colunas obrigatorias: {colunas}.")

    grafo = nx.Graph()
    grafo.add_nodes_from(participantes)
    participantes_validos = set(participantes)

    for linha in pares_df.itertuples(index=False):
        participante_u = linha.participante_u
        participante_v = linha.participante_v
        similaridade = _validar_similaridade(
            linha.sim_final,
            f"{participante_u}-{participante_v}",
        )

        if participante_u not in participantes_validos or participante_v not in participantes_validos:
            raise ValueError("pares_df contem participante que nao esta na lista informada.")

        if participante_u == participante_v:
            continue

        if similaridade >= theta:
            grafo.add_edge(
                participante_u,
                participante_v,
                weight=similaridade,
                distance=1 - similaridade,
            )

    return grafo


def adicionar_atributos_vertices(G, participantes_df=None):
    """
    Adiciona atributos simples aos vertices do grafo.

    Args:
        G: Grafo de similaridade.
        participantes_df: DataFrame opcional com participante_id e rotulo.

    Returns:
        networkx.Graph: O mesmo grafo recebido, possivelmente com atributos nos nos.
    """
    if participantes_df is None:
        return G

    if not isinstance(participantes_df, pd.DataFrame):
        raise ValueError("participantes_df deve ser um pandas.DataFrame.")

    if "participante_id" not in participantes_df.columns:
        raise ValueError("participantes_df deve conter a coluna participante_id.")

    for linha in participantes_df.itertuples(index=False):
        participante_id = getattr(linha, "participante_id")
        if participante_id not in G:
            continue

        if "rotulo" in participantes_df.columns:
            G.nodes[participante_id]["rotulo"] = getattr(linha, "rotulo")

    return G


def obter_jogos_por_rodada(jogos_df):
    """
    Agrupa IDs de jogos por rodada.

    Args:
        jogos_df: DataFrame com as colunas jogo_id e rodada.

    Returns:
        dict: Dicionario no formato {rodada: [jogo_id, ...]}.
    """
    if not isinstance(jogos_df, pd.DataFrame):
        raise ValueError("jogos_df deve ser um pandas.DataFrame.")

    colunas_ausentes = COLUNAS_JOGOS_OBRIGATORIAS.difference(jogos_df.columns)
    if colunas_ausentes:
        colunas = ", ".join(sorted(colunas_ausentes))
        raise ValueError(f"jogos_df sem colunas obrigatorias: {colunas}.")

    jogos_por_rodada = {}
    for rodada, grupo in jogos_df.groupby("rodada", sort=True):
        jogos_por_rodada[rodada] = grupo["jogo_id"].tolist()

    return jogos_por_rodada
