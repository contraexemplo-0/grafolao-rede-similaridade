"""Calculo de componentes, comunidades, cliques e distancias."""

from __future__ import annotations

from collections import Counter

import networkx as nx
import pandas as pd


COLUNAS_DISTANCIAS = ["participante_origem", "participante_destino", "distancia"]


def _ordenar_grupos(grupos):
    return sorted(
        [sorted(grupo) for grupo in grupos],
        key=lambda grupo: (-len(grupo), grupo[0] if grupo else ""),
    )


def obter_componentes_conexas(G):
    """
    Retorna as componentes conexas do grafo em ordem deterministica.

    Args:
        G: Grafo de similaridade.

    Returns:
        list[list[str]]: Componentes com participantes ordenados.
    """
    if G.number_of_nodes() == 0:
        return []

    return _ordenar_grupos(nx.connected_components(G))


def obter_mapa_componentes(G):
    """
    Mapeia cada participante para o identificador numerico de sua componente.

    Args:
        G: Grafo de similaridade.

    Returns:
        dict: Mapa participante_id -> componente.
    """
    mapa = {}
    for indice, componente in enumerate(obter_componentes_conexas(G), start=1):
        for participante in componente:
            mapa[participante] = indice
    return mapa


def obter_maior_componente(G):
    """
    Retorna uma copia do subgrafo induzido pela maior componente conexa.

    Args:
        G: Grafo de similaridade.

    Returns:
        networkx.Graph: Subgrafo da maior componente, ou grafo vazio.
    """
    componentes = obter_componentes_conexas(G)
    if not componentes:
        return G.copy()

    return G.subgraph(componentes[0]).copy()


def detectar_comunidades(G):
    """
    Detecta comunidades por modularidade gulosa ponderada.

    Args:
        G: Grafo de similaridade.

    Returns:
        list[list[str]]: Comunidades com participantes ordenados.
    """
    if G.number_of_nodes() == 0:
        return []

    if G.number_of_edges() == 0:
        return [[participante] for participante in sorted(G.nodes)]

    comunidades = nx.algorithms.community.greedy_modularity_communities(
        G,
        weight="weight",
    )
    return _ordenar_grupos(comunidades)


def obter_mapa_comunidades(G):
    """
    Mapeia cada participante para o identificador numerico de sua comunidade.

    Args:
        G: Grafo de similaridade.

    Returns:
        dict: Mapa participante_id -> comunidade.
    """
    mapa = {}
    for indice, comunidade in enumerate(detectar_comunidades(G), start=1):
        for participante in comunidade:
            mapa[participante] = indice
    return mapa


def listar_cliques_maximais(G):
    """
    Lista cliques maximais de tamanho pelo menos 2.

    Args:
        G: Grafo de similaridade.

    Returns:
        list[list[str]]: Cliques ordenadas por tamanho e lexicograficamente.
    """
    if G.number_of_edges() == 0:
        return []

    cliques = [sorted(clique) for clique in nx.find_cliques(G) if len(clique) >= 2]
    return sorted(cliques, key=lambda clique: (-len(clique), clique))


def contar_participacao_cliques(cliques):
    """
    Conta em quantas cliques cada participante aparece.

    Args:
        cliques (list[list[str]]): Lista de cliques maximais.

    Returns:
        dict: Mapa participante_id -> quantidade de cliques.
    """
    contador = Counter()
    for clique in cliques:
        contador.update(clique)
    return dict(sorted(contador.items()))


def calcular_distancias_maior_componente(G):
    """
    Calcula menores distancias ponderadas na maior componente conexa.

    Args:
        G: Grafo de similaridade.

    Returns:
        pandas.DataFrame: Tabela com origem, destino e distancia.
    """
    maior_componente = obter_maior_componente(G)
    if maior_componente.number_of_nodes() < 2:
        return pd.DataFrame(columns=COLUNAS_DISTANCIAS)

    linhas = []
    distancias = nx.all_pairs_dijkstra_path_length(
        maior_componente,
        weight="distance",
    )

    for origem, destinos in distancias:
        for destino, distancia in destinos.items():
            if origem == destino:
                continue
            linhas.append(
                {
                    "participante_origem": origem,
                    "participante_destino": destino,
                    "distancia": distancia,
                }
            )

    linhas = sorted(
        linhas,
        key=lambda linha: (linha["participante_origem"], linha["participante_destino"]),
    )
    return pd.DataFrame(linhas, columns=COLUNAS_DISTANCIAS)
