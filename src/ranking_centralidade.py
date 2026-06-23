"""Ranking de Centralidade na Rede de Similaridade."""

from __future__ import annotations

from numbers import Real

import networkx as nx
import pandas as pd

from src.config import PAGERANK_DAMPING


COLUNAS_RANKING_CENTRALIDADE = [
    "posicao",
    "participante_id",
    "pagerank",
    "grau",
    "forca",
    "betweenness",
    "comunidade",
]


def _validar_alpha(alpha: float) -> None:
    if not isinstance(alpha, Real) or isinstance(alpha, bool):
        raise ValueError("alpha deve ser um numero no intervalo aberto (0, 1).")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha deve estar no intervalo aberto (0, 1).")


def _validar_pesos_arestas(G) -> None:
    for participante_u, participante_v, dados in G.edges(data=True):
        if "weight" not in dados:
            raise ValueError(
                f"Aresta {participante_u}-{participante_v} sem atributo weight."
            )

        peso = dados["weight"]
        if not isinstance(peso, Real) or isinstance(peso, bool):
            raise ValueError(
                f"Peso da aresta {participante_u}-{participante_v} deve ser numerico."
            )
        if peso < 0 or peso > 1:
            raise ValueError(
                f"Peso da aresta {participante_u}-{participante_v} deve estar entre 0 e 1."
            )


def _mapa_metricas(metricas_df, coluna: str) -> dict:
    if metricas_df is None or coluna not in metricas_df.columns:
        return {}
    if "participante_id" not in metricas_df.columns:
        return {}
    return dict(zip(metricas_df["participante_id"], metricas_df[coluna]))


def calcular_pagerank_ponderado(
    G,
    alpha: float = PAGERANK_DAMPING,
) -> dict:
    """
    Calcula PageRank ponderado pelo atributo de similaridade `weight`.

    Args:
        G: Grafo de similaridade.
        alpha (float): Fator de amortecimento do PageRank.

    Returns:
        dict: Mapa participante_id -> PageRank.
    """
    _validar_alpha(alpha)
    _validar_pesos_arestas(G)

    if G.number_of_nodes() == 0:
        return {}

    return nx.pagerank(G, alpha=alpha, weight="weight")


def gerar_ranking_centralidade(
    G,
    metricas_df=None,
    comunidades=None,
    alpha: float = PAGERANK_DAMPING,
):
    """
    Gera o Ranking de Centralidade na Rede de Similaridade.

    O ranking mede centralidade estrutural na rede de similaridade de palpites,
    nao pontuacao tradicional, originalidade, copia ou influencia causal.

    Args:
        G: Grafo de similaridade.
        metricas_df: DataFrame opcional com metricas individuais auxiliares.
        comunidades: Mapa opcional participante_id -> comunidade.
        alpha (float): Fator de amortecimento do PageRank.

    Returns:
        pandas.DataFrame: Ranking ordenado por PageRank decrescente.
    """
    pagerank = calcular_pagerank_ponderado(G, alpha=alpha)

    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=COLUNAS_RANKING_CENTRALIDADE)

    grau_metricas = _mapa_metricas(metricas_df, "grau")
    forca_metricas = _mapa_metricas(metricas_df, "forca")
    betweenness_metricas = _mapa_metricas(metricas_df, "betweenness")
    comunidade_metricas = _mapa_metricas(metricas_df, "comunidade")

    graus = dict(G.degree())
    forcas = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)

    linhas = []
    for participante in G.nodes:
        comunidade = None
        if participante in comunidade_metricas:
            comunidade = comunidade_metricas[participante]
        elif comunidades is not None:
            comunidade = comunidades.get(participante)

        linhas.append(
            {
                "participante_id": participante,
                "pagerank": pagerank[participante],
                "grau": grau_metricas.get(participante, graus[participante]),
                "forca": forca_metricas.get(participante, forcas[participante]),
                "betweenness": betweenness_metricas.get(
                    participante,
                    betweenness[participante],
                ),
                "comunidade": comunidade,
            }
        )

    ranking_df = pd.DataFrame(linhas)
    ranking_df = ranking_df.sort_values(
        by=["pagerank", "forca", "grau", "participante_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking_df.insert(0, "posicao", range(1, len(ranking_df) + 1))

    return ranking_df[COLUNAS_RANKING_CENTRALIDADE]


def obter_top_k_ranking(ranking_df, k: int = 10):
    """
    Retorna os k primeiros participantes do ranking.

    Args:
        ranking_df: DataFrame de ranking.
        k (int): Quantidade de linhas a retornar.

    Returns:
        pandas.DataFrame: Recorte inicial do ranking.
    """
    if k <= 0:
        raise ValueError("k deve ser maior que zero.")
    return ranking_df.head(k).copy()


def obter_palpiteiro_mais_central(ranking_df):
    """
    Retorna a primeira linha do ranking como dicionario.

    Args:
        ranking_df: DataFrame de ranking.

    Returns:
        dict: Dados do participante mais central, ou dicionario vazio.
    """
    if ranking_df.empty:
        return {}
    return ranking_df.iloc[0].to_dict()
