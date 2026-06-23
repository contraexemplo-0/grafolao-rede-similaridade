"""Analise de sensibilidade do limiar de similaridade."""

from __future__ import annotations

import json
from numbers import Real
from pathlib import Path

import pandas as pd

from src.carregar_dados import carregar_dados
from src.construir_grafo import construir_grafo_similaridade
from src.estruturas_rede import detectar_comunidades, listar_cliques_maximais
from src.exportacao import garantir_diretorios_saida, salvar_tabela
from src.metricas_grafo import calcular_metricas_globais
from src.ranking_centralidade import (
    gerar_ranking_centralidade,
    obter_palpiteiro_mais_central,
)
from src.similaridade import calcular_matriz_similaridade, calcular_pares_similaridade
from src.validacao import validar_dados_entrada, validar_matriz_similaridade


THETAS_PADRAO = [0.20, 0.30, 0.40, 0.50, 0.60]

COLUNAS_ANALISE_LIMIAR = [
    "theta",
    "vertices",
    "arestas",
    "densidade",
    "componentes",
    "maior_componente",
    "grau_medio",
    "forca_media",
    "peso_medio_arestas",
    "agrupamento_medio",
    "quantidade_comunidades",
    "quantidade_cliques_maximais",
    "palpiteiro_mais_central",
    "pagerank_palpiteiro_mais_central",
]


def _validar_thetas(thetas: list[float] | None) -> list[float]:
    valores = THETAS_PADRAO if thetas is None else list(thetas)
    if not valores:
        raise ValueError("A lista de thetas nao pode estar vazia.")

    for theta in valores:
        if not isinstance(theta, Real) or isinstance(theta, bool):
            raise ValueError("Todos os valores de theta devem ser numericos.")
        if theta < 0 or theta > 1:
            raise ValueError("Todos os valores de theta devem estar no intervalo [0, 1].")

    return [float(theta) for theta in valores]


def _mapa_comunidades(comunidades: list[list[str]]) -> dict:
    mapa = {}
    for indice, comunidade in enumerate(comunidades, start=1):
        for participante in comunidade:
            mapa[participante] = indice
    return mapa


def _normalizar_json(valor):
    if isinstance(valor, dict):
        return {str(chave): _normalizar_json(item) for chave, item in valor.items()}
    if isinstance(valor, list):
        return [_normalizar_json(item) for item in valor]
    if hasattr(valor, "item"):
        try:
            return valor.item()
        except ValueError:
            return str(valor)
    return valor


def _salvar_resumo_sensibilidade(resumo: dict, output_dir: str) -> dict:
    resumo_dir = Path(output_dir) / "resumo"
    resumo_dir.mkdir(parents=True, exist_ok=True)

    json_path = resumo_dir / "resumo_sensibilidade_limiar.json"
    txt_path = resumo_dir / "resumo_sensibilidade_limiar.txt"
    resumo_normalizado = _normalizar_json(resumo)

    with json_path.open("w", encoding="utf-8") as arquivo:
        json.dump(resumo_normalizado, arquivo, indent=2, ensure_ascii=True)

    with txt_path.open("w", encoding="utf-8") as arquivo:
        arquivo.write("Resumo da analise de sensibilidade do limiar\n")
        arquivo.write(f"data_dir: {resumo_normalizado['data_dir']}\n")
        arquivo.write(f"output_dir: {resumo_normalizado['output_dir']}\n")
        arquivo.write(f"thetas: {resumo_normalizado['thetas']}\n")
        arquivo.write(f"quantidade_participantes: {resumo_normalizado['quantidade_participantes']}\n")
        arquivo.write(f"quantidade_jogos: {resumo_normalizado['quantidade_jogos']}\n")
        arquivo.write(f"quantidade_pares: {resumo_normalizado['quantidade_pares']}\n")
        arquivo.write("\nResultados por theta:\n")
        for linha in resumo_normalizado["resultados"]:
            arquivo.write(
                "theta={theta}: vertices={vertices}, arestas={arestas}, "
                "componentes={componentes}, maior_componente={maior_componente}, "
                "palpiteiro_mais_central={palpiteiro_mais_central}\n".format(**linha)
            )

    return {
        "resumo_json": str(json_path),
        "resumo_txt": str(txt_path),
    }


def analisar_limiares(
    data_dir: str,
    output_dir: str,
    thetas: list[float] | None = None,
) -> dict:
    """
    Analisa a sensibilidade da rede para diferentes limiares de similaridade.

    Args:
        data_dir (str): Diretorio com CSVs canonicos.
        output_dir (str): Diretorio base de saida.
        thetas (list[float] | None): Limiar(es) no intervalo [0, 1].

    Returns:
        dict: Tabela gerada, resumo e caminhos dos arquivos.
    """
    valores_theta = _validar_thetas(thetas)

    dados = carregar_dados(data_dir)
    validar_dados_entrada(dados)

    participantes = dados["participantes"]["participante_id"].tolist()
    jogos_considerados = dados["jogos"]["jogo_id"].tolist()

    pares_df = calcular_pares_similaridade(
        participantes,
        dados["palpites"],
        jogos_considerados,
    )
    matriz_df = calcular_matriz_similaridade(participantes, pares_df)
    validar_matriz_similaridade(matriz_df)

    linhas = []
    for theta in valores_theta:
        G = construir_grafo_similaridade(participantes, matriz_df, theta=theta)
        metricas = calcular_metricas_globais(G)
        comunidades = detectar_comunidades(G)
        cliques = listar_cliques_maximais(G)
        ranking_df = gerar_ranking_centralidade(
            G,
            comunidades=_mapa_comunidades(comunidades),
        )
        mais_central = obter_palpiteiro_mais_central(ranking_df)

        linhas.append(
            {
                "theta": theta,
                "vertices": metricas["vertices"],
                "arestas": metricas["arestas"],
                "densidade": metricas["densidade"],
                "componentes": metricas["componentes"],
                "maior_componente": metricas["maior_componente"],
                "grau_medio": metricas["grau_medio"],
                "forca_media": metricas["forca_media"],
                "peso_medio_arestas": metricas["peso_medio_arestas"],
                "agrupamento_medio": metricas["agrupamento_medio"],
                "quantidade_comunidades": len(comunidades),
                "quantidade_cliques_maximais": len(cliques),
                "palpiteiro_mais_central": mais_central.get("participante_id"),
                "pagerank_palpiteiro_mais_central": mais_central.get("pagerank"),
            }
        )

    analise_df = pd.DataFrame(linhas, columns=COLUNAS_ANALISE_LIMIAR)

    garantir_diretorios_saida(output_dir)
    tabela_path = Path(output_dir) / "tabelas" / "analise_sensibilidade_limiar.csv"
    salvar_tabela(analise_df, str(tabela_path))

    resumo = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "thetas": valores_theta,
        "quantidade_participantes": len(participantes),
        "quantidade_jogos": len(jogos_considerados),
        "quantidade_pares": len(pares_df),
        "resultados": analise_df.to_dict(orient="records"),
    }
    caminhos_resumo = _salvar_resumo_sensibilidade(resumo, output_dir)

    caminhos = {
        "tabela": str(tabela_path),
        **caminhos_resumo,
    }
    return {
        "tabela": analise_df,
        "resumo": resumo,
        "caminhos": caminhos,
    }

