from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analise_limiar import COLUNAS_ANALISE_LIMIAR, analisar_limiares


def test_analisar_limiares_gera_tabela_e_resumo(tmp_path):
    thetas = [0.2, 0.3, 0.4, 0.5]
    resultado = analisar_limiares(
        data_dir="data/example",
        output_dir=str(tmp_path),
        thetas=thetas,
    )

    tabela_path = tmp_path / "tabelas" / "analise_sensibilidade_limiar.csv"
    resumo_path = tmp_path / "resumo" / "resumo_sensibilidade_limiar.json"

    assert tabela_path.exists()
    assert resumo_path.exists()

    tabela = pd.read_csv(tabela_path)
    assert len(tabela) == len(thetas)
    assert set(COLUNAS_ANALISE_LIMIAR).issubset(tabela.columns)
    assert tabela["theta"].tolist() == pytest.approx(thetas)
    assert resultado["caminhos"]["tabela"] == str(tabela_path)

    resumo = json.loads(resumo_path.read_text(encoding="utf-8"))
    assert resumo["thetas"] == pytest.approx(thetas)
    assert len(resumo["resultados"]) == len(thetas)


def test_vertices_constantes_e_arestas_nao_aumentam_com_theta(tmp_path):
    thetas = [0.2, 0.3, 0.4, 0.5]
    resultado = analisar_limiares(
        data_dir="data/example",
        output_dir=str(tmp_path),
        thetas=thetas,
    )

    tabela = resultado["tabela"]
    assert tabela["vertices"].nunique() == 1

    arestas = tabela["arestas"].tolist()
    assert arestas == sorted(arestas, reverse=True)


@pytest.mark.parametrize(
    "thetas",
    [
        [],
        [-0.1],
        [1.1],
        [0.2, -0.1],
        [0.2, 1.1],
    ],
)
def test_analisar_limiares_rejeita_thetas_invalidos(tmp_path, thetas):
    with pytest.raises(ValueError):
        analisar_limiares(
            data_dir="data/example",
            output_dir=str(tmp_path),
            thetas=thetas,
        )

