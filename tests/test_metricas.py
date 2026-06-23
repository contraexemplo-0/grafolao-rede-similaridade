import networkx as nx
import pytest

from src.estruturas_rede import (
    calcular_distancias_maior_componente,
    contar_participacao_cliques,
    detectar_comunidades,
    listar_cliques_maximais,
    obter_componentes_conexas,
    obter_mapa_comunidades,
)
from src.metricas_grafo import (
    COLUNAS_METRICAS_INDIVIDUAIS,
    calcular_forca_vertices,
    calcular_metricas_globais,
    calcular_metricas_individuais,
)


def adicionar_aresta(G, u, v, weight, distance):
    G.add_edge(u, v, weight=weight, distance=distance)


def test_grafo_caminho_simples_metricas_basicas():
    G = nx.Graph()
    adicionar_aresta(G, "P001", "P002", weight=0.8, distance=0.2)
    adicionar_aresta(G, "P002", "P003", weight=0.6, distance=0.4)

    forcas = calcular_forca_vertices(G)
    metricas_globais = calcular_metricas_globais(G)
    metricas_individuais = calcular_metricas_individuais(G).set_index("participante_id")

    assert dict(G.degree()) == {"P001": 1, "P002": 2, "P003": 1}
    assert forcas["P001"] == pytest.approx(0.8)
    assert forcas["P002"] == pytest.approx(1.4)
    assert forcas["P003"] == pytest.approx(0.6)
    assert metricas_individuais.loc["P002", "betweenness"] > metricas_individuais.loc["P001", "betweenness"]
    assert metricas_individuais.loc["P002", "betweenness"] > metricas_individuais.loc["P003", "betweenness"]
    assert metricas_globais["componentes"] == 1
    assert metricas_globais["maior_componente"] == 3


def test_grafo_com_vertice_isolado():
    G = nx.Graph()
    G.add_nodes_from(["P001", "P002", "P003"])
    adicionar_aresta(G, "P001", "P002", weight=0.7, distance=0.3)

    metricas_globais = calcular_metricas_globais(G)
    metricas_individuais = calcular_metricas_individuais(G).set_index("participante_id")

    assert metricas_globais["componentes"] == 2
    assert metricas_globais["maior_componente"] == 2
    assert "P003" in metricas_individuais.index
    assert metricas_individuais.loc["P003", "grau"] == 0
    assert metricas_individuais.loc["P003", "forca"] == 0


def test_comunidades_sem_arestas():
    G = nx.Graph()
    G.add_nodes_from(["P001", "P002", "P003"])

    comunidades = detectar_comunidades(G)
    mapa_comunidades = obter_mapa_comunidades(G)

    assert comunidades == [["P001"], ["P002"], ["P003"]]
    assert set(mapa_comunidades) == {"P001", "P002", "P003"}
    assert set(mapa_comunidades.values()) == {1, 2, 3}
    assert listar_cliques_maximais(G) == []


def test_cliques_maximais_e_participacao():
    G = nx.Graph()
    adicionar_aresta(G, "P001", "P002", weight=0.9, distance=0.1)
    adicionar_aresta(G, "P001", "P003", weight=0.9, distance=0.1)
    adicionar_aresta(G, "P002", "P003", weight=0.9, distance=0.1)
    adicionar_aresta(G, "P003", "P004", weight=0.6, distance=0.4)

    cliques = listar_cliques_maximais(G)
    participacao = contar_participacao_cliques(cliques)

    assert ["P001", "P002", "P003"] in cliques
    assert ["P003", "P004"] in cliques
    assert participacao["P003"] == 2


def test_distancias_usam_atributo_distance():
    G = nx.Graph()
    adicionar_aresta(G, "P001", "P002", weight=0.9, distance=0.1)
    adicionar_aresta(G, "P002", "P003", weight=0.9, distance=0.1)
    adicionar_aresta(G, "P001", "P003", weight=0.1, distance=0.9)

    distancias = calcular_distancias_maior_componente(G)
    linha = distancias[
        (distancias["participante_origem"] == "P001")
        & (distancias["participante_destino"] == "P003")
    ].iloc[0]

    assert linha["distancia"] == pytest.approx(0.2)


def test_dataframe_metricas_individuais_colunas_pagerank_e_unicidade():
    G = nx.Graph()
    adicionar_aresta(G, "P001", "P002", weight=0.8, distance=0.2)
    adicionar_aresta(G, "P002", "P003", weight=0.6, distance=0.4)

    metricas_df = calcular_metricas_individuais(G)

    assert list(metricas_df.columns) == COLUNAS_METRICAS_INDIVIDUAIS
    assert metricas_df["participante_id"].is_unique
    assert set(metricas_df["participante_id"]) == {"P001", "P002", "P003"}
    assert (metricas_df["pagerank"] >= 0).all()
    assert metricas_df["pagerank"].sum() == pytest.approx(1.0)


def test_grafo_vazio_nao_quebra():
    G = nx.Graph()

    metricas_globais = calcular_metricas_globais(G)
    metricas_individuais = calcular_metricas_individuais(G)

    assert metricas_globais["vertices"] == 0
    assert metricas_globais["arestas"] == 0
    assert metricas_globais["componentes"] == 0
    assert metricas_globais["maior_componente"] == 0
    assert metricas_individuais.empty
    assert list(metricas_individuais.columns) == COLUNAS_METRICAS_INDIVIDUAIS
    assert obter_componentes_conexas(G) == []
    assert detectar_comunidades(G) == []
    assert listar_cliques_maximais(G) == []
