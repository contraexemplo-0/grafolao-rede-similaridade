"""Calculo de metricas globais e individuais do grafo."""

from __future__ import annotations

import networkx as nx
import pandas as pd

from src.config import PAGERANK_DAMPING
from src.estruturas_rede import obter_maior_componente, obter_mapa_comunidades, obter_mapa_componentes


COLUNAS_METRICAS_INDIVIDUAIS = [
    "participante_id",
    "grau",
    "forca",
    "centralidade_grau",
    "betweenness",
    "agrupamento_local",
    "pagerank",
    "componente",
    "comunidade",
]


def calcular_forca_vertices(G):
    """
    Calcula a forca de cada vertice como soma dos pesos incidentes.

    Args:
        G: Grafo de similaridade.

    Returns:
        dict: Mapa participante_id -> forca.
    """
    return dict(G.degree(weight="weight"))


def _media(valores):
    valores = list(valores)
    if not valores:
        return 0
    return sum(valores) / len(valores)


def _distancias_maior_componente(G):
    maior_componente = obter_maior_componente(G)
    if maior_componente.number_of_nodes() < 2:
        return 0, 0

    distancia_media = nx.average_shortest_path_length(
        maior_componente,
        weight="distance",
    )

    distancias = nx.all_pairs_dijkstra_path_length(
        maior_componente,
        weight="distance",
    )
    diametro = max(
        distancia
        for _, destinos in distancias
        for destino, distancia in destinos.items()
        if destino is not None
    )
    return distancia_media, diametro


def calcular_metricas_globais(G):
    """
    Calcula metricas globais do grafo de similaridade.

    Args:
        G: Grafo de similaridade.

    Returns:
        dict: Metricas globais do grafo.
    """
    vertices = G.number_of_nodes()
    arestas = G.number_of_edges()

    if vertices == 0:
        return {
            "vertices": 0,
            "arestas": 0,
            "densidade": 0,
            "componentes": 0,
            "maior_componente": 0,
            "grau_medio": 0,
            "forca_media": 0,
            "peso_medio_arestas": 0,
            "agrupamento_medio": 0,
            "distancia_media_maior_componente": 0,
            "diametro_maior_componente": 0,
        }

    componentes = list(nx.connected_components(G))
    forcas = calcular_forca_vertices(G)
    pesos_arestas = [dados.get("weight", 1) for _, _, dados in G.edges(data=True)]
    distancia_media, diametro = _distancias_maior_componente(G)

    return {
        "vertices": vertices,
        "arestas": arestas,
        "densidade": nx.density(G),
        "componentes": len(componentes),
        "maior_componente": max((len(componente) for componente in componentes), default=0),
        "grau_medio": _media(dict(G.degree()).values()),
        "forca_media": _media(forcas.values()),
        "peso_medio_arestas": _media(pesos_arestas),
        "agrupamento_medio": nx.average_clustering(G, weight="weight"),
        "distancia_media_maior_componente": distancia_media,
        "diametro_maior_componente": diametro,
    }


def calcular_metricas_individuais(G, comunidades=None, componentes=None):
    """
    Calcula metricas individuais dos participantes no grafo.

    Args:
        G: Grafo de similaridade.
        comunidades: Mapa opcional participante_id -> comunidade.
        componentes: Mapa opcional participante_id -> componente.

    Returns:
        pandas.DataFrame: Tabela de metricas individuais.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=COLUNAS_METRICAS_INDIVIDUAIS)

    mapa_componentes = componentes if componentes is not None else obter_mapa_componentes(G)
    mapa_comunidades = comunidades if comunidades is not None else obter_mapa_comunidades(G)

    graus = dict(G.degree())
    forcas = calcular_forca_vertices(G)
    centralidade_grau = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)
    agrupamento_local = nx.clustering(G, weight="weight")
    pagerank = nx.pagerank(G, weight="weight", alpha=PAGERANK_DAMPING)

    linhas = []
    for participante in sorted(G.nodes):
        linhas.append(
            {
                "participante_id": participante,
                "grau": graus[participante],
                "forca": forcas[participante],
                "centralidade_grau": centralidade_grau[participante],
                "betweenness": betweenness[participante],
                "agrupamento_local": agrupamento_local[participante],
                "pagerank": pagerank[participante],
                "componente": mapa_componentes.get(participante),
                "comunidade": mapa_comunidades.get(participante),
            }
        )

    return pd.DataFrame(linhas, columns=COLUNAS_METRICAS_INDIVIDUAIS)


def resumo_metricas_participantes(metricas_df):
    """
    Resume os participantes de maior valor em metricas individuais selecionadas.

    Args:
        metricas_df: DataFrame retornado por calcular_metricas_individuais.

    Returns:
        dict: Participantes de maior grau, forca, betweenness e pagerank.
    """
    if metricas_df.empty:
        return {}

    resumo = {}
    for coluna in ["grau", "forca", "betweenness", "pagerank"]:
        indice = metricas_df[coluna].idxmax()
        resumo[f"maior_{coluna}"] = metricas_df.loc[indice, "participante_id"]

    return resumo
