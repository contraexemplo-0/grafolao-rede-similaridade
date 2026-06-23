"""Exportacao de tabelas, grafos e resumos finais."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import networkx as nx
import pandas as pd


COLUNAS_PARES_GRAFO = {"participante_u", "participante_v", "sim_final"}
COLUNAS_METRICAS_NOS = {
    "participante_id",
    "grau",
    "forca",
    "centralidade_grau",
    "betweenness",
    "agrupamento_local",
    "pagerank",
}


def garantir_diretorios_saida(output_dir: str) -> None:
    """
    Cria os diretorios de saida usados nesta fase.

    Args:
        output_dir (str): Diretorio base de saida.
    """
    base = Path(output_dir)
    for subdir in ["tabelas", "resumo"]:
        (base / subdir).mkdir(parents=True, exist_ok=True)


def salvar_tabela(df, caminho: str) -> None:
    """
    Salva um DataFrame em CSV sem indice.

    Args:
        df: DataFrame a salvar.
        caminho (str): Caminho do arquivo CSV.
    """
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False)


def salvar_matriz_similaridade(matriz_df, caminho: str) -> None:
    """
    Salva a matriz de similaridade preservando o indice.

    Args:
        matriz_df: DataFrame quadrado de similaridade.
        caminho (str): Caminho do arquivo CSV.
    """
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    matriz_df.to_csv(caminho, index=True)


def _normalizar_json(valor):
    if isinstance(valor, dict):
        return {str(chave): _normalizar_json(item) for chave, item in valor.items()}
    if isinstance(valor, list):
        return [_normalizar_json(item) for item in valor]
    if not isinstance(valor, (tuple, set)):
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


def salvar_resumo_execucao(resumo: dict, output_dir: str) -> None:
    """
    Salva resumo de execucao em JSON e TXT.

    Args:
        resumo (dict): Resumo estruturado da execucao.
        output_dir (str): Diretorio base de saida.
    """
    resumo_dir = Path(output_dir) / "resumo"
    resumo_dir.mkdir(parents=True, exist_ok=True)
    resumo_normalizado = _normalizar_json(resumo)

    with (resumo_dir / "resumo_execucao.json").open("w", encoding="utf-8") as arquivo:
        json.dump(resumo_normalizado, arquivo, indent=2, ensure_ascii=True)

    with (resumo_dir / "resumo_execucao.txt").open("w", encoding="utf-8") as arquivo:
        for chave, valor in resumo_normalizado.items():
            arquivo.write(f"{chave}: {valor}\n")


def _carregar_csv(caminho: Path, **kwargs) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo obrigatorio nao encontrado: {caminho}")
    return pd.read_csv(caminho, **kwargs)


def _validar_theta(theta: float) -> None:
    if theta < 0 or theta > 1:
        raise ValueError("theta deve estar no intervalo [0, 1].")


def _validar_colunas(df: pd.DataFrame, colunas: set[str], nome: str) -> None:
    ausentes = colunas.difference(df.columns)
    if ausentes:
        raise ValueError(f"{nome} sem colunas obrigatorias: {', '.join(sorted(ausentes))}.")


def _atributo_json_compativel(valor):
    valor = _normalizar_json(valor)
    if valor is None:
        return None
    if isinstance(valor, (str, int, float, bool)):
        return valor
    return str(valor)


def _atributo_graphml_compativel(valor):
    valor = _normalizar_json(valor)
    if valor is None:
        return ""
    if isinstance(valor, bool):
        return int(valor)
    if isinstance(valor, (str, int, float)):
        return valor
    return str(valor)


def _adicionar_atributos_por_participante(
    G: nx.Graph,
    df: pd.DataFrame,
    chave: str,
    colunas: list[str],
) -> None:
    if df.empty or chave not in df.columns:
        return

    for linha in df.to_dict(orient="records"):
        participante = str(linha[chave])
        if participante not in G:
            G.add_node(participante, participante_id=participante)
        for coluna in colunas:
            if coluna in linha:
                G.nodes[participante][coluna] = _atributo_json_compativel(linha[coluna])


def reconstruir_grafo_final(output_dir: str, theta: float = 0.20) -> nx.Graph:
    """
    Reconstrui o grafo final a partir das tabelas geradas pelo pipeline.

    Args:
        output_dir (str): Diretorio base contendo a pasta `tabelas/`.
        theta (float): Limiar experimental usado para filtrar arestas.

    Returns:
        networkx.Graph: Grafo final com atributos de nos e arestas.
    """
    _validar_theta(theta)
    tabelas_dir = Path(output_dir) / "tabelas"

    pares_df = _carregar_csv(tabelas_dir / "pares_similaridade_final.csv")
    metricas_df = _carregar_csv(tabelas_dir / "metricas_participantes_final.csv")
    componentes_df = _carregar_csv(tabelas_dir / "componentes_final.csv")
    comunidades_df = _carregar_csv(tabelas_dir / "comunidades_final.csv")
    ranking_df = _carregar_csv(tabelas_dir / "ranking_centralidade.csv")

    _validar_colunas(pares_df, COLUNAS_PARES_GRAFO, "pares_similaridade_final")
    _validar_colunas(metricas_df, {"participante_id"}, "metricas_participantes_final")

    G = nx.Graph(theta=float(theta))

    colunas_metricas = [
        coluna
        for coluna in [
            "grau",
            "forca",
            "centralidade_grau",
            "betweenness",
            "agrupamento_local",
            "pagerank",
            "componente",
            "comunidade",
        ]
        if coluna in metricas_df.columns
    ]
    _adicionar_atributos_por_participante(G, metricas_df, "participante_id", colunas_metricas)

    if {"componente", "participante_id"}.issubset(componentes_df.columns):
        _adicionar_atributos_por_participante(
            G,
            componentes_df,
            "participante_id",
            ["componente"],
        )

    if {"comunidade", "participante_id"}.issubset(comunidades_df.columns):
        _adicionar_atributos_por_participante(
            G,
            comunidades_df,
            "participante_id",
            ["comunidade"],
        )

    if {"participante_id", "posicao"}.issubset(ranking_df.columns):
        ranking_aux = ranking_df.rename(columns={"posicao": "posicao_ranking"})
        _adicionar_atributos_por_participante(
            G,
            ranking_aux,
            "participante_id",
            ["posicao_ranking"],
        )

    for linha in pares_df.itertuples(index=False):
        participante_u = str(getattr(linha, "participante_u"))
        participante_v = str(getattr(linha, "participante_v"))
        sim_final = float(getattr(linha, "sim_final"))

        if participante_u not in G:
            G.add_node(participante_u, participante_id=participante_u)
        if participante_v not in G:
            G.add_node(participante_v, participante_id=participante_v)

        if sim_final >= theta:
            G.add_edge(
                participante_u,
                participante_v,
                weight=sim_final,
                distance=1 - sim_final,
            )

    return G


def exportar_grafo_graphml(G, output_path: str) -> None:
    """
    Exporta o grafo em GraphML.

    Args:
        G: Grafo NetworkX.
        output_path (str): Caminho de saida do GraphML.
    """
    caminho = Path(output_path)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    G_export = nx.Graph()
    G_export.graph.update(
        {chave: _atributo_graphml_compativel(valor) for chave, valor in G.graph.items()}
    )
    for no, dados in G.nodes(data=True):
        G_export.add_node(
            str(no),
            **{chave: _atributo_graphml_compativel(valor) for chave, valor in dados.items()},
        )
    for origem, destino, dados in G.edges(data=True):
        G_export.add_edge(
            str(origem),
            str(destino),
            **{chave: _atributo_graphml_compativel(valor) for chave, valor in dados.items()},
        )

    nx.write_graphml(G_export, caminho)


def exportar_grafo_json(G, output_path: str) -> dict:
    """
    Exporta o grafo em JSON estruturado.

    Args:
        G: Grafo NetworkX.
        output_path (str): Caminho de saida do JSON.

    Returns:
        dict: Estrutura exportada.
    """
    caminho = Path(output_path)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    dados = {
        "metadata": {
            "theta": _atributo_json_compativel(G.graph.get("theta")),
            "vertices": int(G.number_of_nodes()),
            "arestas": int(G.number_of_edges()),
        },
        "nodes": [],
        "edges": [],
    }

    for no, atributos in sorted(G.nodes(data=True), key=lambda item: str(item[0])):
        registro = {"id": str(no)}
        for chave, valor in atributos.items():
            if chave != "id":
                registro[chave] = _atributo_json_compativel(valor)
        dados["nodes"].append(registro)

    for origem, destino, atributos in sorted(G.edges(data=True), key=lambda item: (str(item[0]), str(item[1]))):
        registro = {"source": str(origem), "target": str(destino)}
        for chave, valor in atributos.items():
            registro[chave] = _atributo_json_compativel(valor)
        dados["edges"].append(registro)

    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, indent=2, ensure_ascii=True)

    return dados


def _buscar_analise_sensibilidade(output_dir: Path) -> Path | None:
    candidatos = [
        output_dir / "tabelas" / "analise_sensibilidade_limiar.csv",
        Path("outputs") / "tabelas" / "analise_sensibilidade_limiar.csv",
    ]
    for caminho in candidatos:
        if caminho.exists():
            return caminho
    return None


def _linha_unica(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    linha = df.to_dict(orient="records")[0]
    return {chave: _atributo_json_compativel(valor) for chave, valor in linha.items()}


def _formatar_valor(valor) -> str:
    valor = _atributo_json_compativel(valor)
    if isinstance(valor, float):
        return f"{valor:.6g}"
    return str(valor)


def gerar_relatorio_final(output_dir: str, theta: float = 0.20) -> dict:
    """
    Gera relatorio final consolidado em Markdown e JSON.

    Args:
        output_dir (str): Diretorio final do experimento.
        theta (float): Limiar experimental usado no grafo final.

    Returns:
        dict: Relatorio estruturado.
    """
    base = Path(output_dir)
    tabelas_dir = base / "tabelas"
    resumo_dir = base / "resumo"
    figuras_dir = base / "figuras"
    grafos_dir = base / "grafos"
    resumo_dir.mkdir(parents=True, exist_ok=True)

    dados_df = _carregar_csv(tabelas_dir / "resumo_dados_artigo.csv")
    grafo_df = _carregar_csv(tabelas_dir / "resumo_grafo_final.csv")
    componentes_df = _carregar_csv(tabelas_dir / "resumo_componentes_artigo.csv")
    comunidades_df = _carregar_csv(tabelas_dir / "resumo_comunidades_maior_componente.csv")
    ranking_df = _carregar_csv(tabelas_dir / "ranking_centralidade.csv")

    sensibilidade_path = _buscar_analise_sensibilidade(base)
    sensibilidade_df = (
        pd.read_csv(sensibilidade_path) if sensibilidade_path is not None else pd.DataFrame()
    )

    dados = _linha_unica(dados_df)
    grafo = _linha_unica(grafo_df)
    top_ranking = ranking_df.head(10).to_dict(orient="records")

    figuras_finais = [
        "figuras/analise_sensibilidade_limiar.png",
        "figuras/distribuicao_palpites_participante.png",
        "figuras/distribuicao_tamanho_componentes_melhorado.png",
        "figuras/tamanho_comunidades_maior_componente.png",
        "figuras/grafo_maior_componente_comunidades.png",
        "figuras/grafo_maior_componente_filtrado.png",
        "figuras/heatmap_similaridade_ordenado.png",
        "figuras/ranking_centralidade_pagerank_melhorado.png",
    ]
    arquivos_gerados = {
        "graphml": str(grafos_dir / "grafo_final.graphml"),
        "json_grafo": str(grafos_dir / "grafo_final.json"),
        "relatorio_md": str(resumo_dir / "relatorio_final.md"),
        "relatorio_json": str(resumo_dir / "relatorio_final.json"),
        "validacao_txt": str(resumo_dir / "validacao_final.txt"),
        "validacao_json": str(resumo_dir / "validacao_final.json"),
    }

    relatorio = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(base),
        "theta_experimental": float(theta),
        "theta_referencia_estrita": 0.50,
        "dados_experimento": dados,
        "metricas_globais": grafo,
        "componentes": componentes_df.to_dict(orient="records"),
        "comunidades_maior_componente": comunidades_df.to_dict(orient="records"),
        "ranking_top_10": top_ranking,
        "analise_sensibilidade_disponivel": sensibilidade_path is not None,
        "analise_sensibilidade": sensibilidade_df.to_dict(orient="records"),
        "figuras_finais": figuras_finais,
        "arquivos_gerados": arquivos_gerados,
        "interpretacao_critica": [
            "A rede final usa theta = 0.20 como escolha experimental para os dados reais.",
            "O projeto preserva THETA_SIMILARIDADE = 0.50 como referencia estrita e valor padrao.",
            "A rede real ficou fragmentada, com muitas componentes unitarias.",
            "A maior componente contem duas comunidades principais no recorte final.",
            "A fragmentacao esta associada a participacao irregular e a penalizacao por cobertura.",
            "PageRank mede centralidade estrutural na rede de similaridade, nao pontuacao tradicional.",
        ],
        "limitacoes": [
            "A análise usa apenas jogos encerrados com resultado e pelo menos um palpite.",
            "Participantes com poucos palpites tendem a ter baixa cobertura nas comparações.",
            "Comunidades representam estrutura de similaridade, não relações sociais ou causalidade.",
            "O grafo filtrado para visualização não altera as métricas calculadas no grafo final.",
        ],
    }

    linhas = [
        "# Relatório Final  Rede de Similaridade de Palpites",
        "",
        "## 1. Dados do experimento",
    ]
    for chave, valor in dados.items():
        linhas.append(f"- `{chave}`: {_formatar_valor(valor)}")

    linhas.extend(
        [
            "",
            "## 2. Parâmetros da análise",
            f"- Limiar experimental da rede final: `theta = {theta}`.",
            "- Referência estrita preservada no projeto: `THETA_SIMILARIDADE = 0.50`.",
            "- A similaridade entre participantes usa média por jogos comparáveis ajustada por cobertura.",
            "",
            "## 3. Métricas globais do grafo",
        ]
    )
    for chave, valor in grafo.items():
        linhas.append(f"- `{chave}`: {_formatar_valor(valor)}")

    linhas.extend(["", "## 4. Componentes conexas"])
    for linha in componentes_df.to_dict(orient="records"):
        linhas.append(
            "- Tamanho "
            f"{int(linha['tamanho_componente'])}: "
            f"{int(linha['quantidade_componentes'])} componente(s), "
            f"{int(linha['quantidade_vertices_total'])} participante(s)."
        )

    linhas.extend(["", "## 5. Comunidades da maior componente"])
    for linha in comunidades_df.to_dict(orient="records"):
        linhas.append(
            "- Comunidade "
            f"{linha['comunidade']}: {int(linha['tamanho'])} participantes, "
            f"{int(linha['arestas_internas'])} arestas internas, "
            f"participante mais central `{linha['participante_mais_central']}` "
            f"(PageRank {_formatar_valor(linha['pagerank_participante_mais_central'])})."
        )

    linhas.extend(["", "## 6. Ranking de centralidade"])
    for linha in top_ranking[:5]:
        linhas.append(
            f"- {int(linha['posicao'])}. `{linha['participante_id']}` "
            f"PageRank {_formatar_valor(linha['pagerank'])}, "
            f"grau {_formatar_valor(linha['grau'])}, "
            f"força {_formatar_valor(linha['forca'])}."
        )
    linhas.append(
        "- O PageRank representa centralidade estrutural na rede de similaridade, "
        "não pontuação tradicional do bolão."
    )

    linhas.extend(["", "## 7. Figuras finais"])
    for figura in figuras_finais:
        status = "gerada" if (base / figura).exists() else "ausente"
        linhas.append(f"- `{figura}`: {status}.")

    linhas.extend(
        [
            "",
            "## 8. Interpretação crítica",
            "- A rede final usa `theta = 0.20`, escolhido por análise de sensibilidade.",
            "- O modelo preserva `THETA_SIMILARIDADE = 0.50` como referência estrita.",
            "- A rede real ficou fragmentada.",
            "- Há 92 componentes unitárias e uma maior componente com 43 participantes.",
            "- A maior componente contém duas comunidades principais.",
            "- A fragmentação está associada à participação irregular e à penalização por cobertura.",
            "- PageRank mede centralidade estrutural, não pontuação tradicional do bolão.",
            "",
            "## 9. Limitações",
        ]
    )
    for item in relatorio["limitacoes"]:
        linhas.append(f"- {item}")

    linhas.extend(["", "## 10. Arquivos gerados"])
    for nome, caminho in arquivos_gerados.items():
        linhas.append(f"- `{nome}`: `{caminho}`")

    with (resumo_dir / "relatorio_final.md").open("w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(linhas) + "\n")

    relatorio_normalizado = _normalizar_json(relatorio)
    with (resumo_dir / "relatorio_final.json").open("w", encoding="utf-8") as arquivo:
        json.dump(relatorio_normalizado, arquivo, indent=2, ensure_ascii=True)

    return relatorio_normalizado
