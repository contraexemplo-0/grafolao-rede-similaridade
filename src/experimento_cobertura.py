"""Experimento complementar de sensibilidade da penalizacao por cobertura."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

cache_matplotlib = Path(tempfile.gettempdir()) / "matplotlib-cache"
cache_matplotlib.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_matplotlib))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from src.carregar_dados import carregar_dados
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
    exportar_grafo_graphml,
    exportar_grafo_json,
    garantir_diretorios_saida,
    reconstruir_grafo_final,
    salvar_matriz_similaridade,
    salvar_resumo_execucao,
    salvar_tabela,
)
from src.metricas_grafo import calcular_metricas_globais, calcular_metricas_individuais
from src.ranking_centralidade import gerar_ranking_centralidade, obter_palpiteiro_mais_central
from src.similaridade import calcular_matriz_similaridade, calcular_pares_similaridade
from src.validacao import (
    validar_dados_entrada,
    validar_matriz_similaridade,
    validar_ranking_centralidade,
)
from src.visualizacao import (
    gerar_resumo_componentes_artigo,
    gerar_resumo_comunidades_maior_componente,
    gerar_resumo_dados_artigo,
    plotar_distribuicao_palpites_participante,
    plotar_distribuicao_tamanho_componentes,
    plotar_distribuicao_tamanho_componentes_melhorado,
    plotar_grafo_final_comunidades,
    plotar_grafo_maior_componente_comunidades,
    plotar_grafo_maior_componente_filtrado,
    plotar_heatmap_similaridade_ordenado,
    plotar_ranking_centralidade_melhorado,
    plotar_tamanho_comunidades_maior_componente,
    reconstruir_grafo_de_pares,
)


def _validar_gamma(gamma: float) -> float:
    if gamma <= 0:
        raise ValueError("gamma deve ser positivo.")
    return float(gamma)


def _json_safe(valor):
    if isinstance(valor, dict):
        return {str(chave): _json_safe(item) for chave, item in valor.items()}
    if isinstance(valor, list):
        return [_json_safe(item) for item in valor]
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
    linhas = [
        {
            "clique": indice,
            "tamanho": len(clique),
            "participantes": ";".join(clique),
        }
        for indice, clique in enumerate(cliques, start=1)
    ]
    return pd.DataFrame(linhas, columns=["clique", "tamanho", "participantes"])


def _dataframe_participacao_cliques(participantes: list[str], participacao: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "participante_id": participante_id,
                "qtd_cliques": participacao.get(participante_id, 0),
            }
            for participante_id in participantes
        ],
        columns=["participante_id", "qtd_cliques"],
    )


def _salvar_tabelas_experimento(
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


def _componentes_unitarias(componentes_df: pd.DataFrame) -> int:
    if componentes_df.empty:
        return 0
    tamanhos = componentes_df.groupby("componente")["participante_id"].nunique()
    return int((tamanhos == 1).sum())


def _comunidades_na_maior_componente(
    componentes_df: pd.DataFrame,
    comunidades_df: pd.DataFrame,
) -> int:
    if componentes_df.empty or comunidades_df.empty:
        return 0
    tamanhos = componentes_df.groupby("componente")["participante_id"].nunique()
    maior_componente = tamanhos.sort_values(ascending=False).index[0]
    participantes = set(
        componentes_df.loc[
            componentes_df["componente"] == maior_componente,
            "participante_id",
        ]
    )
    return int(
        comunidades_df.loc[
            comunidades_df["participante_id"].isin(participantes),
            "comunidade",
        ].nunique()
    )


def _maior_componente_subgrafo(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G.copy()
    componentes = sorted(
        [sorted(componente) for componente in nx.connected_components(G)],
        key=lambda componente: (-len(componente), componente[0] if componente else ""),
    )
    return G.subgraph(componentes[0]).copy()


def _valor_no(G: nx.Graph, no: str, atributo: str, padrao: float = 0.0) -> float:
    valor = G.nodes[no].get(atributo, padrao)
    if pd.isna(valor):
        return padrao
    return float(valor)


def _comunidade_no(G: nx.Graph, no: str):
    valor = G.nodes[no].get("comunidade", 0)
    if pd.isna(valor):
        return 0
    return valor


def calcular_vertices_ponte_intercomunidades(G: nx.Graph) -> pd.DataFrame:
    """
    Calcula vertices que conectam comunidades diferentes na maior componente.

    Args:
        G: Grafo de similaridade com atributo `comunidade` nos vertices.

    Returns:
        pandas.DataFrame: Tabela ordenada por importancia como ponte.
    """
    maior = _maior_componente_subgrafo(G)
    linhas = []
    for no in maior.nodes:
        comunidade = _comunidade_no(maior, no)
        arestas_inter = [
            (vizinho, dados)
            for vizinho, dados in maior[no].items()
            if _comunidade_no(maior, vizinho) != comunidade
        ]
        grau_inter = len(arestas_inter)
        if grau_inter == 0:
            continue
        forca_inter = sum(float(dados.get("weight", 0.0)) for _, dados in arestas_inter)
        linhas.append(
            {
                "participante_id": no,
                "comunidade": comunidade,
                "pagerank": _valor_no(maior, no, "pagerank", 0.0),
                "betweenness": _valor_no(maior, no, "betweenness", 0.0),
                "grau_intercomunidades": int(grau_inter),
                "forca_intercomunidades": float(forca_inter),
            }
        )

    colunas = [
        "participante_id",
        "comunidade",
        "pagerank",
        "betweenness",
        "grau_intercomunidades",
        "forca_intercomunidades",
    ]
    pontes_df = pd.DataFrame(linhas, columns=colunas)
    if pontes_df.empty:
        return pontes_df
    return pontes_df.sort_values(
        by=["betweenness", "grau_intercomunidades", "forca_intercomunidades", "pagerank"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def construir_metagrafo_comunidades(G: nx.Graph) -> nx.Graph:
    """
    Constroi metagrafo das comunidades da maior componente.

    Cada no representa uma comunidade. As arestas agregam conexoes
    intercomunidades por soma e media dos pesos de similaridade.
    """
    maior = _maior_componente_subgrafo(G)
    meta = nx.Graph()

    comunidades = {}
    for no in maior.nodes:
        comunidade = _comunidade_no(maior, no)
        comunidades.setdefault(comunidade, []).append(no)

    for comunidade, nos in comunidades.items():
        meta.add_node(comunidade, tamanho=len(nos))

    agregados: dict[tuple, list[float]] = {}
    for origem, destino, dados in maior.edges(data=True):
        comunidade_origem = _comunidade_no(maior, origem)
        comunidade_destino = _comunidade_no(maior, destino)
        if comunidade_origem == comunidade_destino:
            continue
        chave = tuple(sorted((comunidade_origem, comunidade_destino)))
        agregados.setdefault(chave, []).append(float(dados.get("weight", 0.0)))

    for (comunidade_a, comunidade_b), pesos in agregados.items():
        peso_total = sum(pesos)
        peso_medio = peso_total / len(pesos) if pesos else 0.0
        meta.add_edge(
            comunidade_a,
            comunidade_b,
            weight=peso_total,
            peso_total=peso_total,
            peso_medio=peso_medio,
            quantidade_arestas=len(pesos),
        )

    return meta


def gerar_resumo_comunidades_apresentacao(
    resumo_comunidades_df: pd.DataFrame,
    vertices_ponte_df: pd.DataFrame,
    output_dir: str,
) -> pd.DataFrame:
    """Gera tabela resumida de comunidades para artigo e apresentacao."""
    linhas = []
    for linha in resumo_comunidades_df.to_dict(orient="records"):
        comunidade = linha["comunidade"]
        pontes = vertices_ponte_df[vertices_ponte_df["comunidade"] == comunidade].head(3)
        linhas.append(
            {
                "comunidade": comunidade,
                "tamanho": int(linha["tamanho"]),
                "participante_mais_central": linha["participante_mais_central"],
                "pagerank_participante_mais_central": linha[
                    "pagerank_participante_mais_central"
                ],
                "vertices_ponte_principais": ";".join(pontes["participante_id"].tolist()),
            }
        )

    resumo_df = pd.DataFrame(
        linhas,
        columns=[
            "comunidade",
            "tamanho",
            "participante_mais_central",
            "pagerank_participante_mais_central",
            "vertices_ponte_principais",
        ],
    )
    caminho = Path(output_dir) / "tabelas" / "resumo_comunidades_apresentacao.csv"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    resumo_df.to_csv(caminho, index=False)
    return resumo_df


def _layout_comunidades_organizado(G: nx.Graph, escala_local: float = 0.32) -> dict:
    """Calcula layout em dois niveis: metagrafo e layouts locais por comunidade."""
    maior = _maior_componente_subgrafo(G)
    meta = construir_metagrafo_comunidades(maior)
    if meta.number_of_nodes() == 0:
        return {}
    if meta.number_of_nodes() == 1:
        pos_meta = {next(iter(meta.nodes)): (0.0, 0.0)}
    else:
        pos_meta = nx.spring_layout(meta, seed=42, weight="weight", scale=2.8)

    posicoes = {}
    comunidades = {}
    for no in maior.nodes:
        comunidades.setdefault(_comunidade_no(maior, no), []).append(no)

    for comunidade, nos in comunidades.items():
        sub = maior.subgraph(nos).copy()
        centro = pos_meta.get(comunidade, (0.0, 0.0))
        if sub.number_of_nodes() == 1:
            no = nos[0]
            posicoes[no] = (float(centro[0]), float(centro[1]))
            continue
        pos_local = nx.spring_layout(sub, seed=42, weight="weight", scale=escala_local)
        for no, pos in pos_local.items():
            posicoes[no] = (float(centro[0] + pos[0]), float(centro[1] + pos[1]))

    return posicoes


def _tamanhos_por_pagerank(G: nx.Graph) -> list[float]:
    valores = [_valor_no(G, no, "pagerank", 0.0) for no in G.nodes]
    maximo = max(valores) if valores else 0.0
    if maximo <= 0:
        return [80.0 for _ in valores]
    return [100.0 + 900.0 * valor / maximo for valor in valores]


def _labels_top_pagerank(G: nx.Graph, top_n: int) -> dict:
    selecionados = sorted(
        G.nodes,
        key=lambda no: (_valor_no(G, no, "pagerank", 0.0), str(no)),
        reverse=True,
    )[:top_n]
    return {no: no for no in selecionados}


def _arestas_intercomunidades_filtradas(
    G: nx.Graph,
    top_por_par_comunidades: int = 12,
    max_total: int | None = None,
) -> list[tuple[str, str]]:
    """Seleciona arestas intercomunidades para visualizacao, sem alterar o grafo."""
    por_par: dict[tuple, list[tuple[str, str]]] = {}
    for origem, destino in G.edges:
        comunidade_origem = _comunidade_no(G, origem)
        comunidade_destino = _comunidade_no(G, destino)
        if comunidade_origem == comunidade_destino:
            continue
        chave = tuple(sorted((comunidade_origem, comunidade_destino)))
        por_par.setdefault(chave, []).append((origem, destino))

    selecionadas = []
    for arestas in por_par.values():
        selecionadas.extend(
            sorted(
                arestas,
                key=lambda aresta: float(G[aresta[0]][aresta[1]].get("weight", 0.0)),
                reverse=True,
            )[:top_por_par_comunidades]
        )

    selecionadas = sorted(
        selecionadas,
        key=lambda aresta: float(G[aresta[0]][aresta[1]].get("weight", 0.0)),
        reverse=True,
    )
    if max_total is not None:
        selecionadas = selecionadas[:max_total]
    return selecionadas


def _largura_aresta(weight: float, minimo: float, escala: float, maximo: float) -> float:
    return min(maximo, minimo + escala * float(weight))


def plotar_grafo_maior_componente_comunidades_organizado(
    G: nx.Graph,
    output_path: str,
    top_labels: int = 8,
    top_inter_por_par_comunidades: int = 12,
) -> None:
    """Plota maior componente com layout em dois niveis por comunidades."""
    maior = _maior_componente_subgrafo(G)
    pos = _layout_comunidades_organizado(maior)
    intra = []
    for origem, destino in maior.edges:
        if _comunidade_no(maior, origem) == _comunidade_no(maior, destino):
            intra.append((origem, destino))
    inter = _arestas_intercomunidades_filtradas(
        maior,
        top_por_par_comunidades=top_inter_por_par_comunidades,
    )

    plt.figure(figsize=(13, 10))
    nx.draw_networkx_edges(
        maior,
        pos,
        edgelist=intra,
        alpha=0.07,
        edge_color="0.6",
        width=[
            _largura_aresta(maior[u][v].get("weight", 0.0), 0.12, 0.7, 0.65)
            for u, v in intra
        ],
    )
    nx.draw_networkx_edges(
        maior,
        pos,
        edgelist=inter,
        alpha=0.28,
        edge_color="0.2",
        width=[
            _largura_aresta(maior[u][v].get("weight", 0.0), 0.22, 1.5, 1.2)
            for u, v in inter
        ],
    )
    nx.draw_networkx_nodes(
        maior,
        pos,
        node_size=_tamanhos_por_pagerank(maior),
        node_color=[int(_comunidade_no(maior, no)) for no in maior.nodes],
        cmap="tab20",
        alpha=0.92,
        linewidths=0.4,
        edgecolors="white",
    )
    nx.draw_networkx_labels(
        maior,
        pos,
        labels=_labels_top_pagerank(maior, top_labels),
        font_size=8,
        bbox={"alpha": 0.68, "edgecolor": "none", "facecolor": "white", "pad": 0.2},
    )
    plt.title(
        "Maior componente conexa organizada por comunidades\n"
        "Filtro visual: apenas conexões intercomunidades mais fortes são destacadas"
    )
    plt.axis("off")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


def plotar_grafo_pontes_intercomunidades(
    G: nx.Graph,
    vertices_ponte_df: pd.DataFrame,
    output_path: str,
    top_labels: int = 8,
    top_edge_labels: int = 6,
    top_inter_edges: int = 40,
) -> None:
    """Plota grafo focado em pontes e arestas intercomunidades."""
    maior = _maior_componente_subgrafo(G)
    pos = _layout_comunidades_organizado(maior)
    inter = _arestas_intercomunidades_filtradas(
        maior,
        top_por_par_comunidades=20,
        max_total=top_inter_edges,
    )
    contexto = []
    for no in set([no for aresta in inter for no in aresta]):
        incidentes_intra = [
            (u, v)
            for u, v in maior.edges(no)
            if _comunidade_no(maior, u) == _comunidade_no(maior, v)
        ]
        contexto.extend(incidentes_intra[:2])
    contexto = list({tuple(sorted(aresta)) for aresta in contexto})

    plt.figure(figsize=(13, 10))
    nx.draw_networkx_edges(
        maior,
        pos,
        edgelist=contexto,
        alpha=0.07,
        edge_color="0.7",
        width=0.45,
    )
    nx.draw_networkx_edges(
        maior,
        pos,
        edgelist=inter,
        alpha=0.48,
        edge_color="0.15",
        width=[
            _largura_aresta(maior[u][v].get("weight", 0.0), 0.45, 2.8, 2.1)
            for u, v in inter
        ],
    )
    nx.draw_networkx_nodes(
        maior,
        pos,
        node_size=_tamanhos_por_pagerank(maior),
        node_color=[int(_comunidade_no(maior, no)) for no in maior.nodes],
        cmap="tab20",
        alpha=0.88,
        linewidths=0.5,
        edgecolors="white",
    )

    pontes = vertices_ponte_df.head(top_labels)["participante_id"].tolist()
    nx.draw_networkx_labels(
        maior,
        pos,
        labels={no: no for no in pontes},
        font_size=8,
        bbox={"alpha": 0.72, "edgecolor": "none", "facecolor": "white", "pad": 0.2},
    )

    top_arestas = inter[:top_edge_labels]
    labels_arestas = {
        aresta: f"{maior[aresta[0]][aresta[1]].get('weight', 0.0):.2f}"
        for aresta in top_arestas
    }
    nx.draw_networkx_edge_labels(
        maior,
        pos,
        edge_labels=labels_arestas,
        font_size=7,
        rotate=False,
        bbox={"alpha": 0.65, "edgecolor": "none", "facecolor": "white"},
    )

    plt.title(
        "Pontes intercomunidades na maior componente\n"
        "Filtro visual: conexões intercomunidades mais fortes"
    )
    plt.axis("off")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


def plotar_metagrafo_comunidades(meta: nx.Graph, output_path: str) -> None:
    """Plota metagrafo das comunidades da maior componente."""
    if meta.number_of_nodes() == 0:
        plt.figure(figsize=(8, 5))
        plt.title("Metagrafo de comunidades")
        plt.axis("off")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=240, bbox_inches="tight")
        plt.close()
        return

    pos = (
        nx.spring_layout(meta, seed=42, weight="weight", scale=2.5)
        if meta.number_of_nodes() > 1
        else {next(iter(meta.nodes)): (0.0, 0.0)}
    )
    tamanhos = [500 + 80 * int(meta.nodes[no].get("tamanho", 1)) for no in meta.nodes]
    larguras = [
        min(5.0, 0.9 + 0.12 * float(dados.get("quantidade_arestas", 0.0)))
        for _, _, dados in meta.edges(data=True)
    ]

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_edges(meta, pos, width=larguras, alpha=0.55, edge_color="0.25")
    nx.draw_networkx_nodes(
        meta,
        pos,
        node_size=tamanhos,
        node_color=list(range(meta.number_of_nodes())),
        cmap="tab20",
        alpha=0.9,
        edgecolors="white",
        linewidths=0.8,
    )
    nx.draw_networkx_labels(
        meta,
        pos,
        labels={
            no: f"Comunidade {no}\n{int(meta.nodes[no].get('tamanho', 0))} participantes"
            for no in meta.nodes
        },
        font_size=9,
    )
    edge_labels = {
        (u, v): f"n={dados.get('quantidade_arestas', 0)} | média={dados.get('peso_medio', 0):.2f}"
        for u, v, dados in meta.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        meta,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        bbox={"alpha": 0.65, "edgecolor": "none", "facecolor": "white"},
    )
    plt.title("Metagrafo de comunidades da maior componente")
    plt.axis("off")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


def _linha_comparacao(
    modelo: str,
    gamma: float,
    theta: float,
    resumo_grafo_df: pd.DataFrame,
    componentes_df: pd.DataFrame,
    comunidades_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> dict:
    metricas = resumo_grafo_df.iloc[0].to_dict()
    participante_mais_central = (
        ranking_df.iloc[0]["participante_id"] if not ranking_df.empty else None
    )
    return {
        "modelo": modelo,
        "gamma": float(gamma),
        "theta": float(theta),
        "vertices": int(metricas["vertices"]),
        "arestas": int(metricas["arestas"]),
        "densidade": float(metricas["densidade"]),
        "componentes": int(metricas["componentes"]),
        "componentes_unitarias": _componentes_unitarias(componentes_df),
        "maior_componente": int(metricas["maior_componente"]),
        "grau_medio": float(metricas["grau_medio"]),
        "peso_medio_arestas": float(metricas["peso_medio_arestas"]),
        "comunidades_maior_componente": _comunidades_na_maior_componente(
            componentes_df,
            comunidades_df,
        ),
        "participante_mais_central": participante_mais_central,
    }


def _carregar_linha_modelo_oficial(
    output_dir: str,
    theta: float,
) -> dict:
    base = Path(output_dir)
    tabelas = base / "tabelas"
    return _linha_comparacao(
        modelo="cobertura_linear",
        gamma=1.0,
        theta=theta,
        resumo_grafo_df=pd.read_csv(tabelas / "resumo_grafo_final.csv"),
        componentes_df=pd.read_csv(tabelas / "componentes_final.csv"),
        comunidades_df=pd.read_csv(tabelas / "comunidades_final.csv"),
        ranking_df=pd.read_csv(tabelas / "ranking_centralidade.csv"),
    )


def _salvar_figura_comparacao(comparacao_df: pd.DataFrame, output_path: str) -> None:
    caminho = Path(output_path)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    metricas = ["arestas", "componentes", "componentes_unitarias", "maior_componente"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metrica in zip(axes.flatten(), metricas):
        barras = ax.bar(comparacao_df["modelo"], comparacao_df[metrica])
        ax.set_title(metrica.replace("_", " "))
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        for barra, valor in zip(barras, comparacao_df[metrica]):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                barra.get_height(),
                str(int(valor)),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("Comparação entre penalização linear e cobertura suavizada")
    fig.tight_layout()
    fig.savefig(caminho, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _gerar_figuras_experimento(output_dir: str, data_dir: str, theta: float) -> dict:
    base = Path(output_dir)
    tabelas = base / "tabelas"
    figuras = base / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)

    componentes_df = pd.read_csv(tabelas / "componentes_final.csv")
    comunidades_df = pd.read_csv(tabelas / "comunidades_final.csv")
    metricas_df = pd.read_csv(tabelas / "metricas_participantes_final.csv")
    pares_df = pd.read_csv(tabelas / "pares_similaridade_final.csv")
    ranking_df = pd.read_csv(tabelas / "ranking_centralidade.csv")
    matriz_df = pd.read_csv(tabelas / "matriz_similaridade_final.csv", index_col=0)
    palpites_df = pd.read_csv(Path(data_dir) / "palpites.csv")

    resumo_componentes_df = gerar_resumo_componentes_artigo(componentes_df, output_dir)
    gerar_resumo_dados_artigo(data_dir, output_dir)

    caminho_palpites = figuras / "distribuicao_palpites_participante.png"
    plotar_distribuicao_palpites_participante(palpites_df, str(caminho_palpites))

    caminho_componentes_basico = figuras / "distribuicao_tamanho_componentes.png"
    plotar_distribuicao_tamanho_componentes(componentes_df, str(caminho_componentes_basico))

    caminho_componentes = figuras / "distribuicao_componentes.png"
    caminho_componentes_melhorado = figuras / "distribuicao_tamanho_componentes_melhorado.png"
    plotar_distribuicao_tamanho_componentes_melhorado(
        resumo_componentes_df,
        str(caminho_componentes),
    )
    plotar_distribuicao_tamanho_componentes_melhorado(
        resumo_componentes_df,
        str(caminho_componentes_melhorado),
    )

    G_vis = reconstruir_grafo_de_pares(
        pares_df,
        metricas_df=metricas_df,
        comunidades_df=comunidades_df,
        theta=theta,
    )
    resumo_comunidades_df = gerar_resumo_comunidades_maior_componente(G_vis, output_dir)
    caminho_comunidades = figuras / "tamanho_comunidades_maior_componente.png"
    plotar_tamanho_comunidades_maior_componente(
        resumo_comunidades_df,
        str(caminho_comunidades),
    )

    vertices_ponte_df = calcular_vertices_ponte_intercomunidades(G_vis)
    caminho_pontes_tabela = tabelas / "vertices_ponte_intercomunidades.csv"
    vertices_ponte_df.to_csv(caminho_pontes_tabela, index=False)
    resumo_apresentacao_df = gerar_resumo_comunidades_apresentacao(
        resumo_comunidades_df,
        vertices_ponte_df,
        output_dir,
    )

    caminho_heatmap = figuras / "heatmap_similaridade_ordenado.png"
    plotar_heatmap_similaridade_ordenado(matriz_df, metricas_df, str(caminho_heatmap))

    caminho_ranking = figuras / "ranking_centralidade_pagerank_melhorado.png"
    plotar_ranking_centralidade_melhorado(ranking_df, str(caminho_ranking))

    caminho_grafo_completo = figuras / "grafo_completo_comunidades.png"
    plotar_grafo_final_comunidades(G_vis, str(caminho_grafo_completo), top_labels=12)

    caminho_maior_componente = figuras / "grafo_maior_componente_comunidades.png"
    plotar_grafo_maior_componente_comunidades(
        G_vis,
        str(caminho_maior_componente),
        top_labels=12,
    )

    caminho_filtrado = figuras / "grafo_maior_componente_filtrado.png"
    plotar_grafo_maior_componente_filtrado(G_vis, str(caminho_filtrado))

    caminho_organizado = figuras / "grafo_maior_componente_comunidades_organizado.png"
    plotar_grafo_maior_componente_comunidades_organizado(
        G_vis,
        str(caminho_organizado),
    )

    caminho_pontes = figuras / "grafo_pontes_intercomunidades.png"
    plotar_grafo_pontes_intercomunidades(
        G_vis,
        vertices_ponte_df,
        str(caminho_pontes),
    )

    meta = construir_metagrafo_comunidades(G_vis)
    caminho_metagrafo = figuras / "metagrafo_comunidades.png"
    plotar_metagrafo_comunidades(meta, str(caminho_metagrafo))

    return {
        "distribuicao_palpites_participante": str(caminho_palpites),
        "distribuicao_tamanho_componentes": str(caminho_componentes_basico),
        "distribuicao_componentes": str(caminho_componentes),
        "distribuicao_tamanho_componentes_melhorado": str(caminho_componentes_melhorado),
        "tamanho_comunidades_maior_componente": str(caminho_comunidades),
        "heatmap_similaridade_ordenado": str(caminho_heatmap),
        "ranking_centralidade_pagerank_melhorado": str(caminho_ranking),
        "grafo_completo_comunidades": str(caminho_grafo_completo),
        "grafo_maior_componente_comunidades": str(caminho_maior_componente),
        "grafo_maior_componente_filtrado": str(caminho_filtrado),
        "grafo_maior_componente_comunidades_organizado": str(caminho_organizado),
        "grafo_pontes_intercomunidades": str(caminho_pontes),
        "metagrafo_comunidades": str(caminho_metagrafo),
        "vertices_ponte_intercomunidades": str(caminho_pontes_tabela),
        "resumo_comunidades_apresentacao": str(
            Path(output_dir) / "tabelas" / "resumo_comunidades_apresentacao.csv"
        ),
        "quantidade_vertices_ponte": int(len(vertices_ponte_df)),
        "quantidade_comunidades_apresentacao": int(len(resumo_apresentacao_df)),
    }


def _gerar_relatorio_modelo_existente(
    output_dir: str,
    theta: float,
    gamma: float,
    modelo: str,
    nome_base: str,
) -> dict:
    """Gera relatorio para um diretorio de resultados ja calculados."""
    base = Path(output_dir)
    tabelas = base / "tabelas"
    resumo_dir = base / "resumo"
    resumo_dir.mkdir(parents=True, exist_ok=True)

    metricas = pd.read_csv(tabelas / "resumo_grafo_final.csv").iloc[0].to_dict()
    comunidades = pd.read_csv(tabelas / "resumo_comunidades_maior_componente.csv")
    pontes = pd.read_csv(tabelas / "vertices_ponte_intercomunidades.csv")
    apresentacao = pd.read_csv(tabelas / "resumo_comunidades_apresentacao.csv")
    ranking = pd.read_csv(tabelas / "ranking_centralidade.csv")

    participante_central = None if ranking.empty else ranking.iloc[0]["participante_id"]
    registro = {
        "parametros": {
            "theta": float(theta),
            "gamma": float(gamma),
            "modelo": modelo,
            "formula": "sim_final = sim_media * cobertura^gamma",
        },
        "metricas_globais": metricas,
        "participante_mais_central": participante_central,
        "comunidades_maior_componente": comunidades.to_dict(orient="records"),
        "vertices_ponte_top_10": pontes.head(10).to_dict(orient="records"),
        "resumo_comunidades_apresentacao": apresentacao.to_dict(orient="records"),
        "observacoes": [
            "Os grafos de nos usam filtro visual de arestas para reduzir poluicao visual.",
            "As metricas e tabelas principais permanecem baseadas no grafo completo.",
            "weight representa similaridade; distance = 1 - weight e usado em caminhos e betweenness.",
        ],
    }

    titulo = (
        "# Relatorio do Modelo Linear Oficial"
        if modelo == "cobertura_linear"
        else "# Relatorio do Experimento de Cobertura"
    )
    linhas = [
        titulo,
        "",
        "## Parametros",
        f"- `theta`: {theta}",
        f"- `gamma`: {gamma}",
        f"- Modelo: `{modelo}`",
        "",
        "## Principais numeros",
        f"- Vertices: {int(metricas['vertices'])}",
        f"- Arestas: {int(metricas['arestas'])}",
        f"- Componentes conexas: {int(metricas['componentes'])}",
        f"- Maior componente: {int(metricas['maior_componente'])}",
        f"- Participante mais central: `{participante_central}`",
        "",
        "## Comunidades da maior componente",
    ]
    for linha in comunidades.to_dict(orient="records"):
        linhas.append(
            f"- Comunidade {linha['comunidade']}: {int(linha['tamanho'])} participantes, "
            f"participante central `{linha['participante_mais_central']}`."
        )

    linhas.extend(["", "## Principais vertices-ponte"])
    for linha in pontes.head(8).to_dict(orient="records"):
        linhas.append(
            f"- `{linha['participante_id']}`: comunidade {linha['comunidade']}, "
            f"grau intercomunidades {int(linha['grau_intercomunidades'])}, "
            f"forca intercomunidades {linha['forca_intercomunidades']:.4f}."
        )

    linhas.extend(
        [
            "",
            "## Observacao visual",
            "- Os grafos de nos usam filtro visual de arestas; isso nao altera metricas, ranking ou tabelas.",
            "- O metagrafo de comunidades e preferivel para artigo e slides quando o grafo completo estiver poluido.",
        ]
    )

    md_path = resumo_dir / f"{nome_base}.md"
    json_path = resumo_dir / f"{nome_base}.json"
    md_path.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    with json_path.open("w", encoding="utf-8") as arquivo:
        json.dump(_json_safe(registro), arquivo, indent=2, ensure_ascii=True)
    return {
        "registro": _json_safe(registro),
        "md": str(md_path),
        "json": str(json_path),
    }


def consolidar_visualizacoes_modelo_existente(
    output_dir: str,
    data_dir: str = "data/processed",
    theta: float = 0.20,
    gamma: float = 1.0,
    modelo: str = "cobertura_linear",
) -> dict:
    """
    Gera artefatos complementares para um modelo ja processado.

    A funcao nao recalcula similaridade, grafo ou ranking. Ela parte dos CSVs
    existentes em `output_dir/tabelas/` e cria tabelas/figuras auxiliares.
    """
    figuras = _gerar_figuras_experimento(output_dir, data_dir, theta)
    nome_relatorio = (
        "relatorio_experimento_modelo_linear"
        if modelo == "cobertura_linear"
        else "relatorio_experimento_cobertura"
    )
    relatorio = _gerar_relatorio_modelo_existente(
        output_dir=output_dir,
        theta=theta,
        gamma=gamma,
        modelo=modelo,
        nome_base=nome_relatorio,
    )
    return {
        "output_dir": output_dir,
        "modelo": modelo,
        "theta": theta,
        "gamma": gamma,
        "figuras": figuras,
        "relatorio": relatorio,
        "caminhos": {
            **figuras,
            f"{nome_relatorio}_md": relatorio["md"],
            f"{nome_relatorio}_json": relatorio["json"],
        },
    }


def gerar_relatorio_experimento_cobertura(
    output_dir: str,
    comparacao_df: pd.DataFrame,
    resumo: dict,
) -> dict:
    """Gera relatorio consolidado do experimento com cobertura suavizada."""
    base = Path(output_dir)
    tabelas = base / "tabelas"
    resumo_dir = base / "resumo"
    resumo_dir.mkdir(parents=True, exist_ok=True)

    metricas = pd.read_csv(tabelas / "resumo_grafo_final.csv").iloc[0].to_dict()
    comunidades = pd.read_csv(tabelas / "resumo_comunidades_maior_componente.csv")
    pontes = pd.read_csv(tabelas / "vertices_ponte_intercomunidades.csv")
    apresentacao = pd.read_csv(tabelas / "resumo_comunidades_apresentacao.csv")

    registro = {
        "parametros": {
            "theta": resumo["theta"],
            "gamma": resumo["gamma"],
            "modelo": resumo["modelo"],
            "formula": "sim_final = sim_media * cobertura^gamma",
        },
        "metricas_globais": metricas,
        "comparacao": comparacao_df.to_dict(orient="records"),
        "comunidades_maior_componente": comunidades.to_dict(orient="records"),
        "vertices_ponte_top_10": pontes.head(10).to_dict(orient="records"),
        "resumo_comunidades_apresentacao": apresentacao.to_dict(orient="records"),
        "interpretacao": [
            "A cobertura suavizada preserva a penalizacao por baixa cobertura, mas reduz sua rigidez.",
            "A topologia fica substancialmente mais conectada do que no modelo linear.",
            "O experimento nao substitui automaticamente o modelo oficial; ele mede sensibilidade metodologica.",
            "weight representa similaridade; distance = 1 - weight e usado para caminhos e betweenness.",
        ],
    }

    linhas = [
        "# Relatório do Experimento Complementar  Cobertura Suavizada",
        "",
        "## Parâmetros",
        f"- `theta`: {resumo['theta']}",
        f"- `gamma`: {resumo['gamma']}",
        "- Fórmula experimental: `sim_final = sim_media * cobertura^gamma`.",
        "- Modelo oficial preservado: cobertura linear com `gamma = 1.0`.",
        "",
        "## Principais números",
        f"- Vértices: {int(metricas['vertices'])}",
        f"- Arestas: {int(metricas['arestas'])}",
        f"- Componentes conexas: {int(metricas['componentes'])}",
        f"- Maior componente: {int(metricas['maior_componente'])}",
        f"- Participante mais central: `{resumo['palpiteiro_mais_central']}`",
        "",
        "## Comparação com o modelo linear",
    ]
    for linha in comparacao_df.to_dict(orient="records"):
        linhas.append(
            f"- `{linha['modelo']}`: {int(linha['arestas'])} arestas, "
            f"{int(linha['componentes'])} componentes, "
            f"{int(linha['componentes_unitarias'])} isolados, "
            f"maior componente com {int(linha['maior_componente'])} participantes."
        )

    linhas.extend(["", "## Comunidades da maior componente"])
    for linha in comunidades.to_dict(orient="records"):
        linhas.append(
            f"- Comunidade {linha['comunidade']}: {int(linha['tamanho'])} participantes, "
            f"participante central `{linha['participante_mais_central']}`."
        )

    linhas.extend(["", "## Principais vértices-ponte"])
    for linha in pontes.head(8).to_dict(orient="records"):
        linhas.append(
            f"- `{linha['participante_id']}`: comunidade {linha['comunidade']}, "
            f"grau intercomunidades {int(linha['grau_intercomunidades'])}, "
            f"força intercomunidades {linha['forca_intercomunidades']:.4f}."
        )

    linhas.extend(
        [
            "",
            "## Interpretação crítica",
            "- A rede suavizada é muito menos fragmentada que a rede com cobertura linear.",
            "- Isso indica que a topologia é sensível à penalização da cobertura.",
            "- O resultado deve ser apresentado como análise complementar, não como substituição automática do modelo principal.",
            "- Nas visualizações, `weight` representa similaridade e controla espessura de arestas; `distance = 1 - weight` é usado em caminhos mínimos e betweenness.",
        ]
    )

    md_path = resumo_dir / "relatorio_experimento_cobertura.md"
    json_path = resumo_dir / "relatorio_experimento_cobertura.json"
    md_path.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    with json_path.open("w", encoding="utf-8") as arquivo:
        json.dump(_json_safe(registro), arquivo, indent=2, ensure_ascii=True)
    return _json_safe(registro)


def executar_experimento_cobertura(
    data_dir: str,
    output_dir: str,
    theta: float = 0.20,
    gamma: float = 0.5,
    modelo_oficial_dir: str = "outputs/final_theta_0_20",
) -> dict:
    """
    Executa experimento complementar com penalizacao por cobertura suavizada.

    Args:
        data_dir (str): Diretorio com CSVs canonicos.
        output_dir (str): Diretorio exclusivo para resultados experimentais.
        theta (float): Limiar de criacao de arestas.
        gamma (float): Expoente aplicado a cobertura.
        modelo_oficial_dir (str): Diretorio dos resultados oficiais para comparacao.

    Returns:
        dict: Objetos principais, caminhos e tabela de comparacao.
    """
    gamma_validado = _validar_gamma(gamma)
    garantir_diretorios_saida(output_dir)
    (Path(output_dir) / "figuras").mkdir(parents=True, exist_ok=True)

    dados = carregar_dados(data_dir)
    validar_dados_entrada(dados)

    participantes = dados["participantes"]["participante_id"].tolist()
    jogos_considerados = dados["jogos"]["jogo_id"].tolist()

    pares_df = calcular_pares_similaridade(
        participantes,
        dados["palpites"],
        jogos_considerados,
        cobertura_gamma=gamma_validado,
    )
    pares_df["cobertura_gamma"] = gamma_validado
    pares_df["cobertura_ajustada"] = pares_df["cobertura"] ** gamma_validado

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
    ranking_df = gerar_ranking_centralidade(grafo, metricas_df=metricas_individuais_df)
    validar_ranking_centralidade(ranking_df)

    componentes_df = _dataframe_componentes(componentes)
    comunidades_df = _dataframe_comunidades(comunidades)
    cliques_df = _dataframe_cliques(cliques)
    participacao_cliques_df = _dataframe_participacao_cliques(
        participantes,
        participacao_cliques,
    )

    caminhos_tabelas = _salvar_tabelas_experimento(
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

    figuras = _gerar_figuras_experimento(output_dir, data_dir, theta)

    grafos_dir = Path(output_dir) / "grafos"
    grafos_dir.mkdir(parents=True, exist_ok=True)
    grafo_exportacao = reconstruir_grafo_final(output_dir, theta=theta)
    graphml_path = grafos_dir / "grafo_final.graphml"
    json_grafo_path = grafos_dir / "grafo_final.json"
    exportar_grafo_graphml(grafo_exportacao, str(graphml_path))
    exportar_grafo_json(grafo_exportacao, str(json_grafo_path))

    palpiteiro_mais_central = obter_palpiteiro_mais_central(ranking_df)
    resumo = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "modelo": "cobertura_suavizada",
        "theta": theta,
        "gamma": gamma_validado,
        "quantidade_participantes": len(participantes),
        "quantidade_jogos": len(jogos_considerados),
        "quantidade_palpites": len(dados["palpites"]),
        "vertices": metricas_globais["vertices"],
        "arestas": metricas_globais["arestas"],
        "densidade": metricas_globais["densidade"],
        "componentes": metricas_globais["componentes"],
        "componentes_unitarias": _componentes_unitarias(componentes_df),
        "maior_componente": metricas_globais["maior_componente"],
        "quantidade_comunidades": len(comunidades),
        "quantidade_cliques_maximais": len(cliques),
        "palpiteiro_mais_central": palpiteiro_mais_central.get("participante_id"),
        "pagerank_palpiteiro_mais_central": palpiteiro_mais_central.get("pagerank"),
    }
    salvar_resumo_execucao(resumo, output_dir)

    comparacao_dir = Path(output_dir).parent
    comparacao_dir.mkdir(parents=True, exist_ok=True)
    linha_oficial = _carregar_linha_modelo_oficial(modelo_oficial_dir, theta)
    linha_experimento = _linha_comparacao(
        modelo="cobertura_suavizada",
        gamma=gamma_validado,
        theta=theta,
        resumo_grafo_df=pd.DataFrame([metricas_globais]),
        componentes_df=componentes_df,
        comunidades_df=comunidades_df,
        ranking_df=ranking_df,
    )
    comparacao_df = pd.DataFrame([linha_oficial, linha_experimento])
    comparacao_csv = comparacao_dir / "comparacao_cobertura.csv"
    comparacao_df.to_csv(comparacao_csv, index=False)
    comparacao_png = comparacao_dir / "comparacao_cobertura.png"
    _salvar_figura_comparacao(comparacao_df, str(comparacao_png))
    comparacao_png_local = Path(output_dir) / "figuras" / "comparacao_cobertura.png"
    _salvar_figura_comparacao(comparacao_df, str(comparacao_png_local))

    relatorio = gerar_relatorio_experimento_cobertura(output_dir, comparacao_df, resumo)

    resumo_comparacao = {
        "comparacao_csv": str(comparacao_csv),
        "comparacao_png": str(comparacao_png),
        "modelos": comparacao_df.to_dict(orient="records"),
    }
    with (Path(output_dir) / "resumo" / "comparacao_cobertura.json").open(
        "w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(resumo_comparacao, arquivo, indent=2, ensure_ascii=True)

    caminhos = {
        **caminhos_tabelas,
        **figuras,
        "comparacao_cobertura_csv": str(comparacao_csv),
        "comparacao_cobertura_png": str(comparacao_png),
        "comparacao_cobertura_png_local": str(comparacao_png_local),
        "grafo_final_graphml": str(graphml_path),
        "grafo_final_json": str(json_grafo_path),
        "resumo_execucao_json": str(Path(output_dir) / "resumo" / "resumo_execucao.json"),
        "resumo_execucao_txt": str(Path(output_dir) / "resumo" / "resumo_execucao.txt"),
        "comparacao_cobertura_json": str(
            Path(output_dir) / "resumo" / "comparacao_cobertura.json"
        ),
        "relatorio_experimento_cobertura_md": str(
            Path(output_dir) / "resumo" / "relatorio_experimento_cobertura.md"
        ),
        "relatorio_experimento_cobertura_json": str(
            Path(output_dir) / "resumo" / "relatorio_experimento_cobertura.json"
        ),
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
        "metricas_globais": metricas_globais,
        "metricas_individuais": metricas_individuais_df,
        "ranking": ranking_df,
        "comparacao": comparacao_df,
        "relatorio": relatorio,
        "resumo": resumo,
        "caminhos": caminhos,
    }
