import pandas as pd
import pytest

from src.similaridade import (
    calcular_matriz_similaridade,
    calcular_pares_similaridade,
    resultado_previsto,
    similaridade_jogo,
    similaridade_participantes,
)


def test_resultado_previsto():
    assert resultado_previsto(2, 1) == "A"
    assert resultado_previsto(1, 1) == "E"
    assert resultado_previsto(0, 2) == "B"


@pytest.mark.parametrize(
    ("palpite_p", "palpite_q", "esperado"),
    [
        ((2, 1), (2, 1), 1.0000),
        ((2, 1), (3, 1), 0.8750),
        ((1, 0), (4, 0), 0.6250),
        ((0, 0), (0, 1), 0.1875),
        ((2, 0), (0, 2), 0.0000),
    ],
)
def test_similaridade_jogo_casos_manuais(palpite_p, palpite_q, esperado):
    assert similaridade_jogo(palpite_p, palpite_q) == pytest.approx(esperado)


def test_similaridade_participantes_com_cobertura_menor_que_um():
    palpites_df = pd.DataFrame(
        [
            ("P001", "J1", 2, 1),
            ("P002", "J1", 2, 1),
            ("P001", "J2", 2, 1),
            ("P002", "J2", 3, 1),
            ("P001", "J3", 0, 0),
            ("P002", "J3", 0, 1),
            ("P001", "J4", 1, 0),
        ],
        columns=[
            "participante_id",
            "jogo_id",
            "gols_a_palpite",
            "gols_b_palpite",
        ],
    )
    jogos_considerados = ["J1", "J2", "J3", "J4"]

    resultado = similaridade_participantes(
        "P001",
        "P002",
        palpites_df,
        jogos_considerados,
    )

    assert resultado["jogos_comparaveis"] == 3
    assert resultado["total_jogos"] == 4
    assert resultado["sim_media"] == pytest.approx(0.6875)
    assert resultado["cobertura"] == pytest.approx(0.75)
    assert resultado["sim_final"] == pytest.approx(0.515625)


def test_similaridade_participantes_sem_jogos_comparaveis():
    palpites_df = pd.DataFrame(
        [
            ("P001", "J1", 2, 1),
            ("P002", "J2", 2, 1),
        ],
        columns=[
            "participante_id",
            "jogo_id",
            "gols_a_palpite",
            "gols_b_palpite",
        ],
    )

    resultado = similaridade_participantes("P001", "P002", palpites_df, ["J1", "J2"])

    assert resultado["jogos_comparaveis"] == 0
    assert resultado["sim_media"] == 0
    assert resultado["cobertura"] == 0
    assert resultado["sim_final"] == 0


def test_calcular_pares_similaridade_gera_pares_unicos_e_colunas():
    participantes = ["P001", "P002", "P003"]
    palpites_df = pd.DataFrame(
        [
            ("P001", "J1", 2, 1),
            ("P002", "J1", 2, 1),
            ("P003", "J1", 0, 1),
            ("P001", "J2", 0, 0),
            ("P002", "J2", 1, 1),
            ("P003", "J2", 0, 0),
        ],
        columns=[
            "participante_id",
            "jogo_id",
            "gols_a_palpite",
            "gols_b_palpite",
        ],
    )

    pares_df = calcular_pares_similaridade(participantes, palpites_df, ["J1", "J2"])

    colunas_esperadas = {
        "participante_u",
        "participante_v",
        "jogos_comparaveis",
        "total_jogos",
        "sim_media",
        "cobertura",
        "sim_final",
    }
    pares = set(zip(pares_df["participante_u"], pares_df["participante_v"]))

    assert len(pares_df) == 3
    assert pares == {("P001", "P002"), ("P001", "P003"), ("P002", "P003")}
    assert all(u != v for u, v in pares)
    assert colunas_esperadas.issubset(pares_df.columns)
    assert pares_df["sim_final"].notna().all()


def test_calcular_matriz_similaridade_quadrada_simetrica_e_ordenada():
    participantes = ["P001", "P002", "P003"]
    pares_df = pd.DataFrame(
        [
            {
                "participante_u": "P001",
                "participante_v": "P002",
                "jogos_comparaveis": 2,
                "total_jogos": 2,
                "sim_media": 0.9,
                "cobertura": 1.0,
                "sim_final": 0.9,
            },
            {
                "participante_u": "P001",
                "participante_v": "P003",
                "jogos_comparaveis": 2,
                "total_jogos": 2,
                "sim_media": 0.25,
                "cobertura": 1.0,
                "sim_final": 0.25,
            },
            {
                "participante_u": "P002",
                "participante_v": "P003",
                "jogos_comparaveis": 2,
                "total_jogos": 2,
                "sim_media": 0.5,
                "cobertura": 1.0,
                "sim_final": 0.5,
            },
        ]
    )

    matriz = calcular_matriz_similaridade(participantes, pares_df)

    assert matriz.shape == (3, 3)
    assert list(matriz.index) == participantes
    assert list(matriz.columns) == participantes
    assert matriz.equals(matriz.T)
    assert all(matriz.loc[p, p] == 1.0 for p in participantes)
    assert matriz.loc["P001", "P002"] == pytest.approx(0.9)
    assert matriz.loc["P002", "P001"] == pytest.approx(0.9)
    assert matriz.loc["P002", "P003"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("palpite_p", "palpite_q"),
    [
        ((-1, 0), (0, 0)),
        ((1, 0, 0), (0, 0)),
        ((1.5, 0), (0, 0)),
    ],
)
def test_similaridade_jogo_rejeita_palpites_invalidos(palpite_p, palpite_q):
    with pytest.raises(ValueError):
        similaridade_jogo(palpite_p, palpite_q)


def test_similaridade_jogo_rejeita_c_invalido():
    with pytest.raises(ValueError):
        similaridade_jogo((1, 0), (1, 0), c=0)


def test_similaridade_participantes_rejeita_lista_de_jogos_vazia():
    palpites_df = pd.DataFrame(
        columns=[
            "participante_id",
            "jogo_id",
            "gols_a_palpite",
            "gols_b_palpite",
        ]
    )

    with pytest.raises(ValueError):
        similaridade_participantes("P001", "P002", palpites_df, [])
