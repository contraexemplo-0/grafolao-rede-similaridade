import networkx as nx
import pandas as pd
import pytest

from src.ranking_centralidade import (
    COLUNAS_RANKING_CENTRALIDADE,
    calcular_pagerank_ponderado,
    gerar_ranking_centralidade,
    obter_palpiteiro_mais_central,
    obter_top_k_ranking,
)


def adicionar_aresta(G, u, v, weight, distance):
    G.add_edge(u, v, weight=weight, distance=distance)


def grafo_caminho():
    G = nx.Graph()
    adicionar_aresta(G, "P001", "P002", weight=0.8, distance=0.2)
    adicionar_aresta(G, "P002", "P003", weight=0.6, distance=0.4)
    return G


def test_pagerank_ponderado_basico():
    G = grafo_caminho()

    pagerank = calcular_pagerank_ponderado(G)

    assert isinstance(pagerank, dict)
    assert set(pagerank) == {"P001", "P002", "P003"}
    assert all(valor >= 0 for valor in pagerank.values())
    assert sum(pagerank.values()) == pytest.approx(1.0)


def test_ranking_contem_colunas_posicoes_e_ordem():
    G = grafo_caminho()

    ranking_df = gerar_ranking_centralidade(G)

    assert list(ranking_df.columns) == COLUNAS_RANKING_CENTRALIDADE
    assert ranking_df["participante_id"].is_unique
    assert ranking_df["posicao"].tolist() == [1, 2, 3]
    assert ranking_df["pagerank"].tolist() == sorted(
        ranking_df["pagerank"].tolist(),
        reverse=True,
    )


def test_ranking_calcula_forca_e_grau():
    G = nx.Graph()
    adicionar_aresta(G, "P001", "P002", weight=0.7, distance=0.3)
    adicionar_aresta(G, "P001", "P003", weight=0.5, distance=0.5)

    ranking_df = gerar_ranking_centralidade(G).set_index("participante_id")

    assert ranking_df.loc["P001", "forca"] == pytest.approx(1.2)
    assert ranking_df.loc["P001", "grau"] == 2


def test_betweenness_usa_distance():
    G = nx.Graph()
    adicionar_aresta(G, "P001", "P002", weight=0.5, distance=0.1)
    adicionar_aresta(G, "P002", "P003", weight=0.5, distance=0.1)
    adicionar_aresta(G, "P001", "P003", weight=0.9, distance=0.9)

    ranking_df = gerar_ranking_centralidade(G).set_index("participante_id")

    assert ranking_df.loc["P002", "betweenness"] > 0


def test_comunidade_a_partir_de_metricas_df():
    G = grafo_caminho()
    metricas_df = pd.DataFrame(
        [
            {"participante_id": "P001", "comunidade": 1},
            {"participante_id": "P002", "comunidade": 1},
            {"participante_id": "P003", "comunidade": 2},
        ]
    )

    ranking_df = gerar_ranking_centralidade(G, metricas_df=metricas_df).set_index(
        "participante_id"
    )

    assert ranking_df.loc["P001", "comunidade"] == 1
    assert ranking_df.loc["P002", "comunidade"] == 1
    assert ranking_df.loc["P003", "comunidade"] == 2


def test_top_k():
    ranking_df = gerar_ranking_centralidade(grafo_caminho())

    assert len(obter_top_k_ranking(ranking_df, k=2)) == 2
    assert len(obter_top_k_ranking(ranking_df, k=10)) == len(ranking_df)

    with pytest.raises(ValueError):
        obter_top_k_ranking(ranking_df, k=0)


def test_palpiteiro_mais_central():
    ranking_df = gerar_ranking_centralidade(grafo_caminho())

    mais_central = obter_palpiteiro_mais_central(ranking_df)

    assert mais_central == ranking_df.iloc[0].to_dict()
    assert obter_palpiteiro_mais_central(pd.DataFrame()) == {}


def test_grafo_vazio():
    G = nx.Graph()

    assert calcular_pagerank_ponderado(G) == {}

    ranking_df = gerar_ranking_centralidade(G)
    assert ranking_df.empty
    assert list(ranking_df.columns) == COLUNAS_RANKING_CENTRALIDADE


@pytest.mark.parametrize("alpha", [0, -0.1, 1, 1.1])
def test_rejeita_alpha_invalido(alpha):
    with pytest.raises(ValueError):
        calcular_pagerank_ponderado(grafo_caminho(), alpha=alpha)


def test_rejeita_aresta_sem_weight():
    G = nx.Graph()
    G.add_edge("P001", "P002", distance=0.2)

    with pytest.raises(ValueError):
        calcular_pagerank_ponderado(G)


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_rejeita_weight_fora_do_intervalo(weight):
    G = nx.Graph()
    adicionar_aresta(G, "P001", "P002", weight=weight, distance=0.2)

    with pytest.raises(ValueError):
        calcular_pagerank_ponderado(G)
