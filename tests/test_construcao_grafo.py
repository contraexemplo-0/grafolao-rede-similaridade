import networkx as nx
import pandas as pd
import pytest

from src.construir_grafo import (
    adicionar_atributos_vertices,
    construir_grafo_de_pares,
    construir_grafo_similaridade,
    obter_jogos_por_rodada,
)


@pytest.fixture
def participantes():
    return ["P001", "P002", "P003"]


@pytest.fixture
def matriz_similaridade(participantes):
    return pd.DataFrame(
        [
            [1.00, 0.70, 0.40],
            [0.70, 1.00, 0.55],
            [0.40, 0.55, 1.00],
        ],
        index=participantes,
        columns=participantes,
    )


@pytest.fixture
def pares_df():
    return pd.DataFrame(
        [
            {"participante_u": "P001", "participante_v": "P002", "sim_final": 0.70},
            {"participante_u": "P001", "participante_v": "P003", "sim_final": 0.40},
            {"participante_u": "P002", "participante_v": "P003", "sim_final": 0.55},
        ]
    )


def test_construir_grafo_similaridade_filtra_por_limiar(participantes, matriz_similaridade):
    grafo = construir_grafo_similaridade(participantes, matriz_similaridade, theta=0.50)

    assert isinstance(grafo, nx.Graph)
    assert not grafo.is_directed()
    assert set(grafo.nodes) == set(participantes)
    assert grafo.has_edge("P001", "P002")
    assert grafo.has_edge("P002", "P003")
    assert not grafo.has_edge("P001", "P003")
    assert grafo["P001"]["P002"]["weight"] == pytest.approx(0.70)
    assert grafo["P001"]["P002"]["distance"] == pytest.approx(0.30)
    assert grafo["P002"]["P003"]["weight"] == pytest.approx(0.55)
    assert grafo["P002"]["P003"]["distance"] == pytest.approx(0.45)


def test_construir_grafo_similaridade_mantem_vertices_isolados(participantes, matriz_similaridade):
    grafo = construir_grafo_similaridade(participantes, matriz_similaridade, theta=0.80)

    assert set(grafo.nodes) == set(participantes)
    assert grafo.number_of_edges() == 0
    assert list(grafo.neighbors("P001")) == []


def test_construir_grafo_de_pares_filtra_por_limiar(participantes, pares_df):
    grafo = construir_grafo_de_pares(participantes, pares_df, theta=0.50)

    assert isinstance(grafo, nx.Graph)
    assert not grafo.is_directed()
    assert set(grafo.nodes) == set(participantes)
    assert grafo.has_edge("P001", "P002")
    assert grafo.has_edge("P002", "P003")
    assert not grafo.has_edge("P001", "P003")
    assert grafo["P001"]["P002"]["weight"] == pytest.approx(0.70)
    assert grafo["P001"]["P002"]["distance"] == pytest.approx(0.30)


@pytest.mark.parametrize("theta", [-0.01, 1.01])
def test_construir_grafo_similaridade_rejeita_theta_invalido(
    participantes,
    matriz_similaridade,
    theta,
):
    with pytest.raises(ValueError):
        construir_grafo_similaridade(participantes, matriz_similaridade, theta=theta)


def test_construir_grafo_similaridade_rejeita_matriz_nao_quadrada(participantes):
    matriz = pd.DataFrame(
        [[1.0, 0.7, 0.4], [0.7, 1.0, 0.5]],
        index=["P001", "P002"],
        columns=participantes,
    )

    with pytest.raises(ValueError):
        construir_grafo_similaridade(participantes, matriz)


def test_construir_grafo_similaridade_rejeita_matriz_sem_participante(
    participantes,
    matriz_similaridade,
):
    matriz = matriz_similaridade.drop(index="P003", columns="P003")

    with pytest.raises(ValueError):
        construir_grafo_similaridade(participantes, matriz)


def test_construir_grafo_similaridade_rejeita_matriz_nao_simetrica(
    participantes,
    matriz_similaridade,
):
    matriz = matriz_similaridade.copy()
    matriz.loc["P001", "P002"] = 0.80

    with pytest.raises(ValueError):
        construir_grafo_similaridade(participantes, matriz)


def test_construir_grafo_similaridade_rejeita_valor_menor_que_zero(
    participantes,
    matriz_similaridade,
):
    matriz = matriz_similaridade.copy()
    matriz.loc["P001", "P002"] = -0.1
    matriz.loc["P002", "P001"] = -0.1

    with pytest.raises(ValueError):
        construir_grafo_similaridade(participantes, matriz)


def test_construir_grafo_similaridade_rejeita_valor_maior_que_um(
    participantes,
    matriz_similaridade,
):
    matriz = matriz_similaridade.copy()
    matriz.loc["P001", "P002"] = 1.1
    matriz.loc["P002", "P001"] = 1.1

    with pytest.raises(ValueError):
        construir_grafo_similaridade(participantes, matriz)


def test_construir_grafo_de_pares_rejeita_colunas_obrigatorias_ausentes(participantes):
    pares = pd.DataFrame([{"participante_u": "P001", "participante_v": "P002"}])

    with pytest.raises(ValueError):
        construir_grafo_de_pares(participantes, pares)


@pytest.mark.parametrize("sim_final", [-0.1, 1.1])
def test_construir_grafo_de_pares_rejeita_sim_final_fora_do_intervalo(
    participantes,
    sim_final,
):
    pares = pd.DataFrame(
        [{"participante_u": "P001", "participante_v": "P002", "sim_final": sim_final}]
    )

    with pytest.raises(ValueError):
        construir_grafo_de_pares(participantes, pares)


def test_adicionar_atributos_vertices_adiciona_rotulo(participantes, matriz_similaridade):
    grafo = construir_grafo_similaridade(participantes, matriz_similaridade)
    participantes_df = pd.DataFrame(
        [
            {"participante_id": "P001", "rotulo": "Participante 001"},
            {"participante_id": "P002", "rotulo": "Participante 002"},
            {"participante_id": "P003", "rotulo": "Participante 003"},
        ]
    )

    grafo = adicionar_atributos_vertices(grafo, participantes_df)

    assert grafo.nodes["P001"]["rotulo"] == "Participante 001"
    assert grafo.nodes["P002"]["rotulo"] == "Participante 002"
    assert grafo.nodes["P003"]["rotulo"] == "Participante 003"


def test_adicionar_atributos_vertices_sem_dataframe_mantem_grafo(
    participantes,
    matriz_similaridade,
):
    grafo = construir_grafo_similaridade(participantes, matriz_similaridade)

    resultado = adicionar_atributos_vertices(grafo)

    assert resultado is grafo


def test_obter_jogos_por_rodada():
    jogos_df = pd.DataFrame(
        [
            {"jogo_id": "J001", "rodada": 1},
            {"jogo_id": "J002", "rodada": 1},
            {"jogo_id": "J003", "rodada": 2},
            {"jogo_id": "J004", "rodada": 2},
        ]
    )

    assert obter_jogos_por_rodada(jogos_df) == {
        1: ["J001", "J002"],
        2: ["J003", "J004"],
    }
