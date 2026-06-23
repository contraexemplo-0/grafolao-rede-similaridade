import json
from pathlib import Path

import pandas as pd
import pytest

from src.carregar_dados import carregar_dados
from src.pipeline import executar_pipeline
from src.validacao import validar_dados_entrada


ARQUIVOS_OBRIGATORIOS = [
    "tabelas/matriz_similaridade_final.csv",
    "tabelas/pares_similaridade_final.csv",
    "tabelas/resumo_grafo_final.csv",
    "tabelas/metricas_participantes_final.csv",
    "tabelas/ranking_centralidade.csv",
    "tabelas/componentes_final.csv",
    "tabelas/comunidades_final.csv",
    "tabelas/cliques_maximais_final.csv",
    "tabelas/participacao_cliques_final.csv",
    "resumo/resumo_execucao.json",
    "resumo/resumo_execucao.txt",
]


def test_executar_pipeline_com_data_example_gera_saidas(tmp_path):
    output_dir = tmp_path / "outputs_example"

    resultado = executar_pipeline(
        data_dir="data/example",
        output_dir=str(output_dir),
        theta=0.5,
    )

    for caminho_relativo in ARQUIVOS_OBRIGATORIOS:
        assert (output_dir / caminho_relativo).exists()

    ranking = pd.read_csv(output_dir / "tabelas/ranking_centralidade.csv")
    participantes = pd.read_csv("data/example/participantes.csv")
    assert len(ranking) == len(participantes)
    assert ranking["pagerank"].tolist() == sorted(
        ranking["pagerank"].tolist(),
        reverse=True,
    )

    matriz = pd.read_csv(
        output_dir / "tabelas/matriz_similaridade_final.csv",
        index_col=0,
    )
    assert matriz.shape[0] == matriz.shape[1]
    assert all(matriz.loc[participante, participante] == pytest.approx(1.0) for participante in matriz.index)

    with (output_dir / "resumo/resumo_execucao.json").open(encoding="utf-8") as arquivo:
        resumo = json.load(arquivo)

    chaves_resumo = {
        "data_dir",
        "output_dir",
        "theta",
        "quantidade_participantes",
        "quantidade_jogos",
        "quantidade_palpites",
        "vertices",
        "arestas",
        "densidade",
        "componentes",
        "maior_componente",
        "quantidade_comunidades",
        "quantidade_cliques_maximais",
        "palpiteiro_mais_central",
        "pagerank_palpiteiro_mais_central",
    }
    assert chaves_resumo.issubset(resumo)
    assert resultado["resumo"]["quantidade_participantes"] == 5


def test_validar_dados_entrada_falha_com_participante_nao_anonimizado():
    dados = carregar_dados("data/example")
    dados["participantes"] = dados["participantes"].copy()
    dados["participantes"].loc[0, "participante_id"] = "Joao"
    dados["palpites"] = dados["palpites"].copy()
    dados["palpites"].loc[
        dados["palpites"]["participante_id"] == "P001",
        "participante_id",
    ] = "Joao"

    with pytest.raises(ValueError):
        validar_dados_entrada(dados)


def test_validar_dados_entrada_falha_com_palpite_para_jogo_inexistente():
    dados = carregar_dados("data/example")
    dados["palpites"] = dados["palpites"].copy()
    dados["palpites"].loc[0, "jogo_id"] = "J999"

    with pytest.raises(ValueError):
        validar_dados_entrada(dados)
