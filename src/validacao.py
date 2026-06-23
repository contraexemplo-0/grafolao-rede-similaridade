"""Validacoes de dados, matriz, grafo, ranking e saidas."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


COLUNAS_ENTRADA = {
    "participantes": {"participante_id", "rotulo"},
    "jogos": {"jogo_id", "rodada"},
    "palpites": {
        "participante_id",
        "jogo_id",
        "gols_a_palpite",
        "gols_b_palpite",
    },
    "resultados": {"jogo_id", "gols_a_real", "gols_b_real", "status"},
}

COLUNAS_RANKING = {
    "posicao",
    "participante_id",
    "pagerank",
    "grau",
    "forca",
    "betweenness",
    "comunidade",
}

PADRAO_PARTICIPANTE_ANONIMO = re.compile(r"^P\d{3,}$")
PADRAO_EMAIL = re.compile(r"[\w.\-+]+@[\w.\-]+\.[A-Za-z]{2,}")
COLUNAS_SENSIVEIS = {
    "nome",
    "name",
    "email",
    "avatarurl",
    "avatar_url",
    "googleid",
    "google_id",
    "usuarioid",
    "usuario_id",
    "userid",
    "user_id",
    "token",
    "hash",
    "password",
    "senha",
}


def _validar_colunas(nome: str, df: pd.DataFrame, colunas: set[str]) -> None:
    ausentes = colunas.difference(df.columns)
    if ausentes:
        colunas_ausentes = ", ".join(sorted(ausentes))
        raise ValueError(f"{nome} sem colunas obrigatorias: {colunas_ausentes}.")


def _validar_sem_negativos(df: pd.DataFrame, colunas: list[str], nome: str) -> None:
    for coluna in colunas:
        if (df[coluna] < 0).any():
            raise ValueError(f"{nome} contem placares negativos na coluna {coluna}.")


def validar_dados_entrada(dados: dict) -> None:
    """
    Valida os DataFrames canonicos de entrada.

    Args:
        dados (dict): Dicionario retornado por carregar_dados.
    """
    chaves_ausentes = set(COLUNAS_ENTRADA).difference(dados)
    if chaves_ausentes:
        chaves = ", ".join(sorted(chaves_ausentes))
        raise ValueError(f"Dados de entrada incompletos: {chaves}.")

    for nome, colunas in COLUNAS_ENTRADA.items():
        _validar_colunas(nome, dados[nome], colunas)

    participantes = dados["participantes"]
    jogos = dados["jogos"]
    palpites = dados["palpites"]
    resultados = dados["resultados"]

    ids_participantes = set(participantes["participante_id"])
    ids_jogos = set(jogos["jogo_id"])

    participantes_invalidos = [
        participante_id
        for participante_id in ids_participantes
        if not PADRAO_PARTICIPANTE_ANONIMO.match(str(participante_id))
    ]
    if participantes_invalidos:
        raise ValueError("Participantes devem estar anonimizados no padrao P001, P002, ...")

    palpites_participantes_invalidos = set(palpites["participante_id"]).difference(
        ids_participantes
    )
    if palpites_participantes_invalidos:
        ids = ", ".join(sorted(palpites_participantes_invalidos))
        raise ValueError(f"Palpites referenciam participantes inexistentes: {ids}.")

    palpites_jogos_invalidos = set(palpites["jogo_id"]).difference(ids_jogos)
    if palpites_jogos_invalidos:
        ids = ", ".join(sorted(palpites_jogos_invalidos))
        raise ValueError(f"Palpites referenciam jogos inexistentes: {ids}.")

    resultados_jogos_invalidos = set(resultados["jogo_id"]).difference(ids_jogos)
    if resultados_jogos_invalidos:
        ids = ", ".join(sorted(resultados_jogos_invalidos))
        raise ValueError(f"Resultados referenciam jogos inexistentes: {ids}.")

    _validar_sem_negativos(
        palpites,
        ["gols_a_palpite", "gols_b_palpite"],
        "palpites",
    )
    _validar_sem_negativos(
        resultados,
        ["gols_a_real", "gols_b_real"],
        "resultados",
    )


def validar_matriz_similaridade(matriz_df) -> None:
    """
    Valida formato e valores da matriz de similaridade.

    Args:
        matriz_df: DataFrame quadrado de similaridade.
    """
    if not isinstance(matriz_df, pd.DataFrame):
        raise ValueError("Matriz de similaridade deve ser um pandas.DataFrame.")
    if matriz_df.shape[0] != matriz_df.shape[1]:
        raise ValueError("Matriz de similaridade deve ser quadrada.")
    if list(matriz_df.index) != list(matriz_df.columns):
        raise ValueError("Indice e colunas da matriz devem conter os mesmos participantes.")
    if ((matriz_df < 0) | (matriz_df > 1)).to_numpy().any():
        raise ValueError("Valores de similaridade devem estar entre 0 e 1.")

    diferenca_simetria = (matriz_df - matriz_df.T).abs().to_numpy().max()
    if diferenca_simetria > 1e-9:
        raise ValueError("Matriz de similaridade deve ser simetrica.")

    for participante in matriz_df.index:
        if abs(float(matriz_df.loc[participante, participante]) - 1.0) > 1e-9:
            raise ValueError("Diagonal da matriz de similaridade deve ser igual a 1.")


def validar_ranking_centralidade(ranking_df) -> None:
    """
    Valida o Ranking de Centralidade na Rede de Similaridade.

    Args:
        ranking_df: DataFrame de ranking.
    """
    _validar_colunas("ranking_centralidade", ranking_df, COLUNAS_RANKING)

    if ranking_df.empty:
        return

    if ranking_df["participante_id"].duplicated().any():
        raise ValueError("Ranking contem participantes repetidos.")

    posicoes_esperadas = list(range(1, len(ranking_df) + 1))
    if ranking_df["posicao"].tolist() != posicoes_esperadas:
        raise ValueError("Posicoes do ranking devem comecar em 1 e ser consecutivas.")

    if (ranking_df["pagerank"] < 0).any():
        raise ValueError("Ranking contem PageRank negativo.")

    pageranks = ranking_df["pagerank"].tolist()
    if pageranks != sorted(pageranks, reverse=True):
        raise ValueError("Ranking deve estar ordenado por PageRank decrescente.")

    if abs(float(ranking_df["pagerank"].sum()) - 1.0) > 1e-6:
        raise ValueError("Soma dos PageRanks deve ser aproximadamente 1.")


def _normalizar_json(valor):
    if isinstance(valor, dict):
        return {str(chave): _normalizar_json(item) for chave, item in valor.items()}
    if isinstance(valor, list):
        return [_normalizar_json(item) for item in valor]
    try:
        if pd.isna(valor):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(valor, "item"):
        try:
            return valor.item()
        except ValueError:
            return str(valor)
    return valor


def _registrar_validacao(resultado: dict, nome: str, ok: bool, mensagem: str) -> None:
    resultado["checks"].append({"nome": nome, "ok": bool(ok), "mensagem": mensagem})
    if not ok:
        resultado["ok"] = False


def _arquivo_relativo(output_dir: Path, caminho_relativo: str) -> Path:
    return output_dir / Path(caminho_relativo)


def _verificar_arquivos_essenciais(output_dir: Path, resultado: dict) -> None:
    arquivos = [
        "tabelas/resumo_dados_artigo.csv",
        "tabelas/resumo_grafo_final.csv",
        "tabelas/resumo_componentes_artigo.csv",
        "tabelas/resumo_comunidades_maior_componente.csv",
        "tabelas/ranking_centralidade.csv",
        "tabelas/metricas_participantes_final.csv",
        "tabelas/pares_similaridade_final.csv",
        "tabelas/matriz_similaridade_final.csv",
        "figuras/analise_sensibilidade_limiar.png",
        "figuras/distribuicao_palpites_participante.png",
        "figuras/distribuicao_tamanho_componentes_melhorado.png",
        "figuras/tamanho_comunidades_maior_componente.png",
        "figuras/ranking_centralidade_pagerank_melhorado.png",
        "figuras/heatmap_similaridade_ordenado.png",
        "grafos/grafo_final.graphml",
        "grafos/grafo_final.json",
        "resumo/relatorio_final.md",
        "resumo/relatorio_final.json",
    ]

    for arquivo in arquivos:
        caminho = _arquivo_relativo(output_dir, arquivo)
        ok = caminho.exists() and caminho.stat().st_size > 0
        mensagem = "arquivo encontrado" if ok else "arquivo ausente ou vazio"
        _registrar_validacao(resultado, f"arquivo:{arquivo}", ok, mensagem)


def _verificar_consistencia_grafo_json(output_dir: Path, resultado: dict) -> None:
    grafo_json = output_dir / "grafos" / "grafo_final.json"
    resumo_csv = output_dir / "tabelas" / "resumo_grafo_final.csv"
    if not grafo_json.exists() or not resumo_csv.exists():
        _registrar_validacao(
            resultado,
            "consistencia_grafo_json",
            False,
            "grafo_final.json ou resumo_grafo_final.csv ausente",
        )
        return

    with grafo_json.open("r", encoding="utf-8") as arquivo:
        grafo = json.load(arquivo)
    resumo = pd.read_csv(resumo_csv)
    if resumo.empty:
        _registrar_validacao(
            resultado,
            "consistencia_grafo_json",
            False,
            "resumo_grafo_final.csv vazio",
        )
        return

    vertices_json = int(grafo.get("metadata", {}).get("vertices", -1))
    arestas_json = int(grafo.get("metadata", {}).get("arestas", -1))
    vertices_csv = int(resumo.iloc[0]["vertices"])
    arestas_csv = int(resumo.iloc[0]["arestas"])

    ok = vertices_json == vertices_csv and arestas_json == arestas_csv
    _registrar_validacao(
        resultado,
        "consistencia_grafo_json",
        ok,
        f"json=({vertices_json}, {arestas_json}), csv=({vertices_csv}, {arestas_csv})",
    )


def _verificar_ranking(output_dir: Path, resultado: dict) -> None:
    caminho = output_dir / "tabelas" / "ranking_centralidade.csv"
    if not caminho.exists():
        _registrar_validacao(resultado, "ranking", False, "ranking_centralidade.csv ausente")
        return

    ranking = pd.read_csv(caminho)
    try:
        validar_ranking_centralidade(ranking)
        _registrar_validacao(resultado, "ranking", True, "ranking valido")
    except ValueError as erro:
        _registrar_validacao(resultado, "ranking", False, str(erro))


def _verificar_dados_sensiveis(output_dir: Path, resultado: dict) -> None:
    tabelas_dir = output_dir / "tabelas"
    if not tabelas_dir.exists():
        _registrar_validacao(resultado, "dados_sensiveis", False, "diretorio de tabelas ausente")
        return

    problemas: list[str] = []
    for caminho in sorted(tabelas_dir.glob("*.csv")):
        try:
            df = pd.read_csv(caminho, nrows=50)
        except pd.errors.EmptyDataError:
            continue

        colunas_normalizadas = {str(coluna).lower().replace("-", "_") for coluna in df.columns}
        colunas_sensiveis = sorted(colunas_normalizadas.intersection(COLUNAS_SENSIVEIS))
        if colunas_sensiveis:
            problemas.append(f"{caminho.name}: colunas sensiveis {', '.join(colunas_sensiveis)}")

        texto = caminho.read_text(encoding="utf-8", errors="ignore")
        if PADRAO_EMAIL.search(texto):
            problemas.append(f"{caminho.name}: padrao de email encontrado")

    _registrar_validacao(
        resultado,
        "dados_sensiveis",
        not problemas,
        "sem colunas sensiveis ou emails" if not problemas else "; ".join(problemas),
    )


def gerar_validacao_final(output_dir: str) -> dict:
    """
    Gera validacao final dos artefatos exportados.

    Args:
        output_dir (str): Diretorio final do experimento.

    Returns:
        dict: Resultado estruturado da validacao.
    """
    base = Path(output_dir)
    resumo_dir = base / "resumo"
    resumo_dir.mkdir(parents=True, exist_ok=True)

    resultado = {
        "ok": True,
        "output_dir": str(base),
        "checks": [],
    }

    _verificar_arquivos_essenciais(base, resultado)
    _verificar_consistencia_grafo_json(base, resultado)
    _verificar_ranking(base, resultado)
    _verificar_dados_sensiveis(base, resultado)

    resultado = _normalizar_json(resultado)
    with (resumo_dir / "validacao_final.json").open("w", encoding="utf-8") as arquivo:
        json.dump(resultado, arquivo, indent=2, ensure_ascii=True)

    with (resumo_dir / "validacao_final.txt").open("w", encoding="utf-8") as arquivo:
        status = "OK" if resultado["ok"] else "FALHA"
        arquivo.write(f"validacao_final: {status}\n")
        arquivo.write(f"output_dir: {resultado['output_dir']}\n")
        for check in resultado["checks"]:
            marcador = "OK" if check["ok"] else "FALHA"
            arquivo.write(f"[{marcador}] {check['nome']}: {check['mensagem']}\n")

    return resultado
