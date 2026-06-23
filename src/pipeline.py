"""Orquestracao do pipeline completo."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.carregar_dados import carregar_dados
from src.config import THETA_SIMILARIDADE
from src.construir_grafo import adicionar_atributos_vertices, construir_grafo_similaridade
from src.estruturas_rede import (
    contar_participacao_cliques,
    detectar_comunidades,
    listar_cliques_maximais,
    obter_componentes_conexas,
    obter_mapa_comunidades,
    obter_mapa_componentes,
)
from src.exportacao import (
    garantir_diretorios_saida,
    salvar_matriz_similaridade,
    salvar_resumo_execucao,
    salvar_tabela,
)
from src.metricas_grafo import calcular_metricas_globais, calcular_metricas_individuais
from src.ranking_centralidade import (
    gerar_ranking_centralidade,
    obter_palpiteiro_mais_central,
)
from src.similaridade import calcular_matriz_similaridade, calcular_pares_similaridade
from src.validacao import (
    validar_dados_entrada,
    validar_matriz_similaridade,
    validar_ranking_centralidade,
)


def _dataframe_componentes(componentes: list[list[str]]) -> pd.DataFrame:
    linhas = []
    for indice, componente in enumerate(componentes, start=1):
        for participante_id in componente:
            linhas.append({"componente": indice, "participante_id": participante_id})
    return pd.DataFrame(linhas, columns=["componente", "participante_id"])


def _dataframe_comunidades(comunidades: list[list[str]]) -> pd.DataFrame:
    linhas = []
    for indice, comunidade in enumerate(comunidades, start=1):
        for participante_id in comunidade:
            linhas.append({"comunidade": indice, "participante_id": participante_id})
    return pd.DataFrame(linhas, columns=["comunidade", "participante_id"])


def _dataframe_cliques(cliques: list[list[str]]) -> pd.DataFrame:
    linhas = []
    for indice, clique in enumerate(cliques, start=1):
        linhas.append(
            {
                "clique": indice,
                "tamanho": len(clique),
                "participantes": ";".join(clique),
            }
        )
    return pd.DataFrame(linhas, columns=["clique", "tamanho", "participantes"])


def _dataframe_participacao_cliques(
    participantes: list[str],
    participacao_cliques: dict,
) -> pd.DataFrame:
    linhas = [
        {
            "participante_id": participante_id,
            "qtd_cliques": participacao_cliques.get(participante_id, 0),
        }
        for participante_id in participantes
    ]
    return pd.DataFrame(linhas, columns=["participante_id", "qtd_cliques"])


def _salvar_tabelas(
    output_dir: str,
    matriz_similaridade,
    pares_df,
    metricas_globais,
    metricas_individuais_df,
    ranking_df,
    componentes_df,
    comunidades_df,
    cliques_df,
    participacao_cliques_df,
) -> dict:
    tabelas_dir = Path(output_dir) / "tabelas"
    caminhos = {
        "matriz_similaridade_final": tabelas_dir / "matriz_similaridade_final.csv",
        "pares_similaridade_final": tabelas_dir / "pares_similaridade_final.csv",
        "resumo_grafo_final": tabelas_dir / "resumo_grafo_final.csv",
        "metricas_participantes_final": tabelas_dir / "metricas_participantes_final.csv",
        "ranking_centralidade": tabelas_dir / "ranking_centralidade.csv",
        "componentes_final": tabelas_dir / "componentes_final.csv",
        "comunidades_final": tabelas_dir / "comunidades_final.csv",
        "cliques_maximais_final": tabelas_dir / "cliques_maximais_final.csv",
        "participacao_cliques_final": tabelas_dir / "participacao_cliques_final.csv",
    }

    salvar_matriz_similaridade(matriz_similaridade, str(caminhos["matriz_similaridade_final"]))
    salvar_tabela(pares_df, str(caminhos["pares_similaridade_final"]))
    salvar_tabela(pd.DataFrame([metricas_globais]), str(caminhos["resumo_grafo_final"]))
    salvar_tabela(metricas_individuais_df, str(caminhos["metricas_participantes_final"]))
    salvar_tabela(ranking_df, str(caminhos["ranking_centralidade"]))
    salvar_tabela(componentes_df, str(caminhos["componentes_final"]))
    salvar_tabela(comunidades_df, str(caminhos["comunidades_final"]))
    salvar_tabela(cliques_df, str(caminhos["cliques_maximais_final"]))
    salvar_tabela(participacao_cliques_df, str(caminhos["participacao_cliques_final"]))

    return {nome: str(caminho) for nome, caminho in caminhos.items()}


def executar_pipeline(
    data_dir: str,
    output_dir: str,
    theta: float = THETA_SIMILARIDADE,
) -> dict:
    """
    Executa o pipeline principal usando os CSVs canonicos.

    Args:
        data_dir (str): Diretorio com os CSVs canonicos.
        output_dir (str): Diretorio base para arquivos gerados.
        theta (float): Limiar de similaridade para criacao de arestas.

    Returns:
        dict: Objetos principais e caminhos gerados.
    """
    garantir_diretorios_saida(output_dir)

    dados = carregar_dados(data_dir)
    validar_dados_entrada(dados)

    participantes = dados["participantes"]["participante_id"].tolist()
    jogos_considerados = dados["jogos"]["jogo_id"].tolist()

    pares_df = calcular_pares_similaridade(
        participantes,
        dados["palpites"],
        jogos_considerados,
    )
    matriz_similaridade = calcular_matriz_similaridade(participantes, pares_df)
    validar_matriz_similaridade(matriz_similaridade)

    grafo = construir_grafo_similaridade(participantes, matriz_similaridade, theta=theta)
    adicionar_atributos_vertices(grafo, dados["participantes"])

    componentes = obter_componentes_conexas(grafo)
    mapa_componentes = obter_mapa_componentes(grafo)
    comunidades = detectar_comunidades(grafo)
    mapa_comunidades = obter_mapa_comunidades(grafo)
    cliques = listar_cliques_maximais(grafo)
    participacao_cliques = contar_participacao_cliques(cliques)

    metricas_globais = calcular_metricas_globais(grafo)
    metricas_individuais_df = calcular_metricas_individuais(
        grafo,
        comunidades=mapa_comunidades,
        componentes=mapa_componentes,
    )
    ranking_df = gerar_ranking_centralidade(
        grafo,
        metricas_df=metricas_individuais_df,
    )
    validar_ranking_centralidade(ranking_df)

    componentes_df = _dataframe_componentes(componentes)
    comunidades_df = _dataframe_comunidades(comunidades)
    cliques_df = _dataframe_cliques(cliques)
    participacao_cliques_df = _dataframe_participacao_cliques(
        participantes,
        participacao_cliques,
    )

    caminhos_tabelas = _salvar_tabelas(
        output_dir,
        matriz_similaridade,
        pares_df,
        metricas_globais,
        metricas_individuais_df,
        ranking_df,
        componentes_df,
        comunidades_df,
        cliques_df,
        participacao_cliques_df,
    )

    palpiteiro_mais_central = obter_palpiteiro_mais_central(ranking_df)
    resumo = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "theta": theta,
        "quantidade_participantes": len(participantes),
        "quantidade_jogos": len(jogos_considerados),
        "quantidade_palpites": len(dados["palpites"]),
        "vertices": metricas_globais["vertices"],
        "arestas": metricas_globais["arestas"],
        "densidade": metricas_globais["densidade"],
        "componentes": metricas_globais["componentes"],
        "maior_componente": metricas_globais["maior_componente"],
        "quantidade_comunidades": len(comunidades),
        "quantidade_cliques_maximais": len(cliques),
        "palpiteiro_mais_central": palpiteiro_mais_central.get("participante_id"),
        "pagerank_palpiteiro_mais_central": palpiteiro_mais_central.get("pagerank"),
    }
    salvar_resumo_execucao(resumo, output_dir)

    caminhos = {
        **caminhos_tabelas,
        "resumo_execucao_json": str(Path(output_dir) / "resumo" / "resumo_execucao.json"),
        "resumo_execucao_txt": str(Path(output_dir) / "resumo" / "resumo_execucao.txt"),
    }

    return {
        "dados": dados,
        "participantes": participantes,
        "jogos_considerados": jogos_considerados,
        "pares_similaridade": pares_df,
        "matriz_similaridade": matriz_similaridade,
        "grafo": grafo,
        "componentes": componentes,
        "comunidades": comunidades,
        "cliques": cliques,
        "metricas_globais": metricas_globais,
        "metricas_individuais": metricas_individuais_df,
        "ranking": ranking_df,
        "resumo": resumo,
        "caminhos": caminhos,
    }
