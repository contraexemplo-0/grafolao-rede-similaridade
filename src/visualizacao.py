"""Geracao de visualizacoes em PNG com matplotlib."""

from __future__ import annotations

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


COLUNAS_PARES_VISUALIZACAO = {"participante_u", "participante_v", "sim_final"}


def _validar_theta(theta: float) -> None:
    if theta < 0 or theta > 1:
        raise ValueError("theta deve estar no intervalo [0, 1].")


def _validar_colunas(df: pd.DataFrame, colunas: set[str], nome: str) -> None:
    ausentes = colunas.difference(df.columns)
    if ausentes:
        raise ValueError(f"{nome} sem colunas obrigatorias: {', '.join(sorted(ausentes))}.")


def _salvar_figura(output_path: str, dpi: int = 240) -> None:
    caminho = Path(output_path)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(caminho, dpi=dpi, bbox_inches="tight")
    plt.close()


def _valor_no(G, no: str, atributo: str, padrao: float = 0.0) -> float:
    valor = G.nodes[no].get(atributo, padrao)
    if pd.isna(valor):
        return padrao
    return float(valor)


def _ranking_nos_para_labels(G, top_labels: int) -> dict:
    if top_labels <= 0:
        return {}

    atributo = "pagerank" if any("pagerank" in dados for _, dados in G.nodes(data=True)) else "grau"
    ordenados = sorted(
        G.nodes,
        key=lambda no: (_valor_no(G, no, atributo), G.degree(no), str(no)),
        reverse=True,
    )
    selecionados = set(ordenados[:top_labels])
    return {no: no for no in G.nodes if no in selecionados}


def _tamanhos_nos(G, atributo_preferido: str = "pagerank") -> list[float]:
    if G.number_of_nodes() == 0:
        return []

    if any(atributo_preferido in dados for _, dados in G.nodes(data=True)):
        valores = [_valor_no(G, no, atributo_preferido) for no in G.nodes]
    else:
        valores = [float(G.degree(no)) for no in G.nodes]

    maximo = max(valores) if valores else 0
    if maximo <= 0:
        return [80.0 for _ in valores]
    return [80.0 + 900.0 * (valor / maximo) for valor in valores]


def _larguras_arestas(G) -> list[float]:
    return [0.4 + 2.8 * float(dados.get("weight", 0.0)) for _, _, dados in G.edges(data=True)]


def _layout_grafo(G):
    if G.number_of_nodes() == 0:
        return {}
    return nx.spring_layout(G, seed=42, weight="weight", iterations=80)


def _adicionar_atributos_metricas(G, metricas_df: pd.DataFrame | None) -> None:
    if metricas_df is None or metricas_df.empty:
        return
    if "participante_id" not in metricas_df.columns:
        return

    atributos = ["pagerank", "grau", "forca", "betweenness", "comunidade", "componente"]
    for linha in metricas_df.to_dict(orient="records"):
        participante = linha["participante_id"]
        if participante not in G:
            G.add_node(participante)
        for atributo in atributos:
            if atributo in linha and not pd.isna(linha[atributo]):
                G.nodes[participante][atributo] = linha[atributo]


def _adicionar_comunidades(G, comunidades_df: pd.DataFrame | None) -> None:
    if comunidades_df is None or comunidades_df.empty:
        return
    if not {"participante_id", "comunidade"}.issubset(comunidades_df.columns):
        return

    for linha in comunidades_df.to_dict(orient="records"):
        participante = linha["participante_id"]
        if participante not in G:
            G.add_node(participante)
        G.nodes[participante]["comunidade"] = linha["comunidade"]


def reconstruir_grafo_de_pares(
    pares_df,
    metricas_df=None,
    comunidades_df=None,
    theta: float = 0.20,
):
    """
    Reconstrui o grafo de similaridade a partir da tabela de pares.

    Args:
        pares_df: DataFrame com pares e `sim_final`.
        metricas_df: DataFrame opcional com atributos dos vertices.
        comunidades_df: DataFrame opcional com comunidades.
        theta (float): Limiar de criacao de arestas.

    Returns:
        networkx.Graph: Grafo reconstruido para visualizacao.
    """
    _validar_theta(theta)
    _validar_colunas(pares_df, COLUNAS_PARES_VISUALIZACAO, "pares_df")

    G = nx.Graph()
    _adicionar_atributos_metricas(G, metricas_df)
    _adicionar_comunidades(G, comunidades_df)

    for linha in pares_df.itertuples(index=False):
        participante_u = getattr(linha, "participante_u")
        participante_v = getattr(linha, "participante_v")
        sim_final = float(getattr(linha, "sim_final"))

        if participante_u not in G:
            G.add_node(participante_u)
        if participante_v not in G:
            G.add_node(participante_v)
        if sim_final >= theta:
            G.add_edge(
                participante_u,
                participante_v,
                weight=sim_final,
                distance=1 - sim_final,
            )

    return G


def plotar_grafo_final_similaridade(
    G,
    output_path: str,
    top_labels: int = 15,
):
    """Gera a figura do grafo final de similaridade."""
    plt.figure(figsize=(13, 10))
    pos = _layout_grafo(G)
    tamanhos = _tamanhos_nos(G, "pagerank")

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.22,
        width=_larguras_arestas(G),
        edge_color="0.45",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=tamanhos,
        node_color=[G.degree(no) for no in G.nodes],
        cmap="viridis",
        alpha=0.88,
        linewidths=0.3,
        edgecolors="white",
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels=_ranking_nos_para_labels(G, top_labels),
        font_size=8,
        font_color="black",
    )

    plt.title("Grafo completo de similaridade entre participantes")
    plt.axis("off")
    _salvar_figura(output_path)


def plotar_grafo_final_comunidades(
    G,
    output_path: str,
    top_labels: int = 15,
):
    """Gera a figura principal do grafo colorido por comunidades."""
    plt.figure(figsize=(13, 10))
    pos = _layout_grafo(G)
    tamanhos = _tamanhos_nos(G, "pagerank")
    comunidades = [
        int(_valor_no(G, no, "comunidade", 0)) if "comunidade" in G.nodes[no] else 0
        for no in G.nodes
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.18,
        width=_larguras_arestas(G),
        edge_color="0.45",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=tamanhos,
        node_color=comunidades,
        cmap="tab20",
        alpha=0.9,
        linewidths=0.3,
        edgecolors="white",
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels=_ranking_nos_para_labels(G, top_labels),
        font_size=8,
        font_color="black",
    )

    plt.title("Grafo completo de similaridade por comunidades")
    plt.axis("off")
    _salvar_figura(output_path)


def plotar_heatmap_similaridade(
    matriz_df,
    output_path: str,
    max_participantes: int = 80,
    participantes_ordenados: list[str] | None = None,
):
    """Gera heatmap da matriz de similaridade."""
    if participantes_ordenados:
        participantes = [p for p in participantes_ordenados if p in matriz_df.index]
    else:
        participantes = list(matriz_df.index)
    participantes = participantes[:max_participantes]
    matriz = matriz_df.loc[participantes, participantes]

    fig, ax = plt.subplots(figsize=(11, 9))
    imagem = ax.imshow(matriz.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    fig.colorbar(imagem, ax=ax, label="Similaridade")

    if len(participantes) <= 40:
        ax.set_xticks(range(len(participantes)))
        ax.set_yticks(range(len(participantes)))
        ax.set_xticklabels(participantes, rotation=90, fontsize=7)
        ax.set_yticklabels(participantes, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title("Heatmap da matriz de similaridade")
    ax.set_xlabel("Participantes")
    ax.set_ylabel("Participantes")
    _salvar_figura(output_path)


def plotar_ranking_centralidade(
    ranking_df,
    output_path: str,
    top_k: int = 15,
):
    """Gera grafico de barras do ranking por PageRank."""
    ranking = ranking_df.sort_values("pagerank", ascending=False).head(top_k).copy()
    ranking = ranking.iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(ranking["participante_id"], ranking["pagerank"])
    plt.title(f"Top {min(top_k, len(ranking_df))} participantes por centralidade PageRank")
    plt.xlabel("PageRank")
    plt.ylabel("Participante")
    _salvar_figura(output_path)


def plotar_analise_sensibilidade_limiar(
    sensibilidade_df,
    output_path: str,
):
    """Gera grafico de sensibilidade do limiar theta."""
    plt.figure(figsize=(10, 7))
    plt.plot(sensibilidade_df["theta"], sensibilidade_df["arestas"], marker="o", label="Arestas")
    plt.plot(
        sensibilidade_df["theta"],
        sensibilidade_df["componentes"],
        marker="o",
        label="Componentes conexas",
    )
    plt.plot(
        sensibilidade_df["theta"],
        sensibilidade_df["maior_componente"],
        marker="o",
        label="Maior componente conexa",
    )
    plt.title("Sensibilidade da rede ao limiar θ")
    plt.xlabel("Limiar θ")
    plt.ylabel("Quantidade")
    plt.legend()
    plt.grid(alpha=0.25)
    _salvar_figura(output_path)


def plotar_distribuicao_palpites_participante(
    palpites_df,
    output_path: str,
):
    """Gera distribuicao da quantidade de palpites por participante."""
    _validar_colunas(palpites_df, {"participante_id", "jogo_id"}, "palpites_df")
    contagem_palpites = palpites_df.groupby("participante_id")["jogo_id"].nunique()
    distribuicao = contagem_palpites.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(distribuicao.index.astype(str), distribuicao.values)
    ax.set_title("Distribuição de palpites válidos por participante")
    ax.set_xlabel("Quantidade de palpites válidos")
    ax.set_ylabel("Quantidade de participantes")
    ax.grid(axis="y", alpha=0.25)
    _salvar_figura(output_path)


def plotar_distribuicao_tamanho_componentes(
    componentes_df,
    output_path: str,
):
    """Gera distribuicao dos tamanhos das componentes conexas."""
    _validar_colunas(componentes_df, {"componente", "participante_id"}, "componentes_df")
    tamanhos = componentes_df.groupby("componente")["participante_id"].nunique()
    distribuicao = tamanhos.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(distribuicao.index.astype(str), distribuicao.values)
    ax.set_title("Distribuição dos tamanhos das componentes conexas")
    ax.set_xlabel("Tamanho da componente conexa")
    ax.set_ylabel("Quantidade de componentes conexas")
    ax.grid(axis="y", alpha=0.25)
    _salvar_figura(output_path)


def extrair_maior_componente(G):
    """Retorna uma copia do subgrafo induzido pela maior componente conexa."""
    if G.number_of_nodes() == 0:
        return G.copy()

    componentes = sorted(
        [sorted(componente) for componente in nx.connected_components(G)],
        key=lambda componente: (-len(componente), componente[0] if componente else ""),
    )
    return G.subgraph(componentes[0]).copy()


def plotar_grafo_maior_componente_comunidades(
    G,
    output_path: str,
    top_labels: int = 12,
):
    """Gera grafo da maior componente conexa colorido por comunidades."""
    subgrafo = extrair_maior_componente(G)

    plt.figure(figsize=(12, 9))
    pos = _layout_grafo(subgrafo)
    tamanhos = _tamanhos_nos(subgrafo, "pagerank")
    comunidades = [
        int(_valor_no(subgrafo, no, "comunidade", 0))
        if "comunidade" in subgrafo.nodes[no]
        else 0
        for no in subgrafo.nodes
    ]

    nx.draw_networkx_edges(
        subgrafo,
        pos,
        alpha=0.25,
        width=_larguras_arestas(subgrafo),
        edge_color="0.45",
    )
    nx.draw_networkx_nodes(
        subgrafo,
        pos,
        node_size=tamanhos,
        node_color=comunidades,
        cmap="tab20",
        alpha=0.92,
        linewidths=0.4,
        edgecolors="white",
    )
    nx.draw_networkx_labels(
        subgrafo,
        pos,
        labels=_ranking_nos_para_labels(subgrafo, top_labels),
        font_size=8,
        font_color="black",
    )

    plt.title("Maior componente conexa por comunidades")
    plt.axis("off")
    _salvar_figura(output_path)


def _ordenar_participantes_heatmap(matriz_df, metricas_df, max_participantes: int) -> list[str]:
    metricas = metricas_df[metricas_df["participante_id"].isin(matriz_df.index)].copy()
    for coluna, padrao in [
        ("componente", 0),
        ("comunidade", 0),
        ("pagerank", 0.0),
    ]:
        if coluna not in metricas.columns:
            metricas[coluna] = padrao

    metricas["componente"] = pd.to_numeric(metricas["componente"], errors="coerce").fillna(0)
    metricas["comunidade"] = pd.to_numeric(metricas["comunidade"], errors="coerce").fillna(0)
    metricas["pagerank"] = pd.to_numeric(metricas["pagerank"], errors="coerce").fillna(0.0)

    if len(metricas) > max_participantes and "componente" in metricas.columns:
        maior_componente = metricas["componente"].value_counts().idxmax()
        metricas_maior = metricas[metricas["componente"] == maior_componente].copy()
        metricas_resto = metricas[metricas["componente"] != maior_componente].copy()

        metricas_maior = metricas_maior.sort_values(
            by=["comunidade", "pagerank", "participante_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        selecionadas = metricas_maior.head(max_participantes)
        vagas = max_participantes - len(selecionadas)
        if vagas > 0:
            metricas_resto = metricas_resto.sort_values(
                by=["pagerank", "participante_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            selecionadas = pd.concat([selecionadas, metricas_resto.head(vagas)])
        metricas = selecionadas

    metricas = metricas.sort_values(
        by=["componente", "comunidade", "pagerank", "participante_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    return metricas["participante_id"].tolist()


def plotar_heatmap_similaridade_ordenado(
    matriz_df,
    metricas_df,
    output_path: str,
    max_participantes: int = 100,
):
    """Gera heatmap da matriz de similaridade ordenado por estrutura da rede."""
    _validar_colunas(metricas_df, {"participante_id"}, "metricas_df")
    participantes = _ordenar_participantes_heatmap(matriz_df, metricas_df, max_participantes)
    matriz = matriz_df.loc[participantes, participantes]

    fig, ax = plt.subplots(figsize=(11, 9))
    imagem = ax.imshow(matriz.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    fig.colorbar(imagem, ax=ax, label="Similaridade")

    if len(participantes) <= 45:
        ax.set_xticks(range(len(participantes)))
        ax.set_yticks(range(len(participantes)))
        ax.set_xticklabels(participantes, rotation=90, fontsize=7)
        ax.set_yticklabels(participantes, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title("Heatmap de similaridade ordenado por componente conexa e comunidade")
    ax.set_xlabel("Participantes")
    ax.set_ylabel("Participantes")
    _salvar_figura(output_path)


def plotar_ranking_centralidade_melhorado(
    ranking_df,
    output_path: str,
    top_k: int = 15,
):
    """Gera grafico de barras do ranking PageRank com valores anotados."""
    ranking = ranking_df.sort_values("pagerank", ascending=False).head(top_k).copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    barras = ax.barh(ranking["participante_id"], ranking["pagerank"])
    ax.invert_yaxis()
    ax.set_title("Top participantes por centralidade PageRank")
    ax.set_xlabel("PageRank")
    ax.set_ylabel("Participante")

    maximo = float(ranking["pagerank"].max()) if not ranking.empty else 0.0
    deslocamento = maximo * 0.01 if maximo > 0 else 0.001
    for barra, valor in zip(barras, ranking["pagerank"]):
        ax.text(
            barra.get_width() + deslocamento,
            barra.get_y() + barra.get_height() / 2,
            f"{valor:.4f}",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(right=maximo * 1.14 if maximo > 0 else 1)
    _salvar_figura(output_path)


def gerar_resumo_dados_artigo(
    data_dir: str,
    output_dir: str,
) -> pd.DataFrame:
    """Gera tabela de caracterizacao dos dados usados no artigo."""
    base_dados = Path(data_dir)
    participantes = _carregar_csv(base_dados / "participantes.csv")
    jogos = _carregar_csv(base_dados / "jogos.csv")
    palpites = _carregar_csv(base_dados / "palpites.csv")
    resultados = _carregar_csv(base_dados / "resultados.csv")

    _validar_colunas(participantes, {"participante_id"}, "participantes")
    _validar_colunas(palpites, {"participante_id", "jogo_id"}, "palpites")

    contagem = (
        palpites.groupby("participante_id")["jogo_id"]
        .nunique()
        .reindex(participantes["participante_id"], fill_value=0)
    )
    linha = {
        "participantes_validos": int(len(participantes)),
        "jogos_validos": int(len(jogos)),
        "palpites_validos": int(len(palpites)),
        "resultados_validos": int(len(resultados)),
        "media_palpites_por_participante": float(contagem.mean()) if len(contagem) else 0,
        "mediana_palpites_por_participante": float(contagem.median()) if len(contagem) else 0,
        "min_palpites_por_participante": int(contagem.min()) if len(contagem) else 0,
        "max_palpites_por_participante": int(contagem.max()) if len(contagem) else 0,
        "participantes_com_1_palpite": int((contagem == 1).sum()),
        "participantes_com_ate_4_palpites": int((contagem <= 4).sum()),
    }
    resumo_df = pd.DataFrame([linha])

    caminho = Path(output_dir) / "tabelas" / "resumo_dados_artigo.csv"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    resumo_df.to_csv(caminho, index=False)
    return resumo_df


def gerar_resumo_componentes_artigo(
    componentes_df,
    output_dir: str,
) -> pd.DataFrame:
    """Gera tabela agregada dos tamanhos das componentes."""
    _validar_colunas(componentes_df, {"componente", "participante_id"}, "componentes_df")
    tamanhos = componentes_df.groupby("componente")["participante_id"].nunique()
    distribuicao = tamanhos.value_counts().sort_index()

    resumo_df = pd.DataFrame(
        [
            {
                "tamanho_componente": int(tamanho),
                "quantidade_componentes": int(quantidade),
                "quantidade_vertices_total": int(tamanho * quantidade),
            }
            for tamanho, quantidade in distribuicao.items()
        ]
    )

    caminho = Path(output_dir) / "tabelas" / "resumo_componentes_artigo.csv"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    resumo_df.to_csv(caminho, index=False)
    return resumo_df


def gerar_resumo_comunidades_maior_componente(
    G,
    output_dir: str,
) -> pd.DataFrame:
    """Gera resumo das comunidades presentes na maior componente conexa."""
    maior = extrair_maior_componente(G)
    linhas = []

    comunidades = {}
    for no, dados in maior.nodes(data=True):
        comunidade = dados.get("comunidade")
        if pd.isna(comunidade):
            comunidade = 0
        comunidades.setdefault(comunidade, []).append(no)

    for comunidade, nos in comunidades.items():
        subgrafo = maior.subgraph(nos).copy()
        pageranks = {no: _valor_no(maior, no, "pagerank", 0.0) for no in nos}
        forcas = [_valor_no(maior, no, "forca", 0.0) for no in nos]
        participante_central = max(nos, key=lambda no: (pageranks[no], str(no)))

        linhas.append(
            {
                "comunidade": comunidade,
                "tamanho": int(len(nos)),
                "arestas_internas": int(subgrafo.number_of_edges()),
                "densidade_interna": nx.density(subgrafo) if len(nos) > 1 else 0,
                "forca_media": sum(forcas) / len(forcas) if forcas else 0,
                "pagerank_total": sum(pageranks.values()),
                "participante_mais_central": participante_central,
                "pagerank_participante_mais_central": pageranks[participante_central],
            }
        )

    resumo_df = pd.DataFrame(linhas)
    if not resumo_df.empty:
        resumo_df = resumo_df.sort_values(
            by=["tamanho", "pagerank_total", "comunidade"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)

    caminho = Path(output_dir) / "tabelas" / "resumo_comunidades_maior_componente.csv"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    resumo_df.to_csv(caminho, index=False)
    return resumo_df


def plotar_distribuicao_tamanho_componentes_melhorado(
    resumo_componentes_df,
    output_path: str,
):
    """Gera distribuicao de componentes com valores nas barras."""
    _validar_colunas(
        resumo_componentes_df,
        {"tamanho_componente", "quantidade_componentes"},
        "resumo_componentes_df",
    )

    df = resumo_componentes_df.sort_values("tamanho_componente")
    fig, ax = plt.subplots(figsize=(11, 6))
    barras = ax.bar(df["tamanho_componente"].astype(str), df["quantidade_componentes"])
    ax.set_title("Distribuição dos tamanhos das componentes conexas")
    ax.set_xlabel("Tamanho da componente conexa")
    ax.set_ylabel("Quantidade de componentes conexas")
    ax.grid(axis="y", alpha=0.25)

    for barra, valor in zip(barras, df["quantidade_componentes"]):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height(),
            str(int(valor)),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _salvar_figura(output_path)


def plotar_tamanho_comunidades_maior_componente(
    resumo_comunidades_df,
    output_path: str,
):
    """Gera grafico de tamanhos das comunidades da maior componente."""
    _validar_colunas(resumo_comunidades_df, {"comunidade", "tamanho"}, "resumo_comunidades_df")
    df = resumo_comunidades_df.sort_values(
        by=["tamanho", "comunidade"],
        ascending=[False, True],
        kind="mergesort",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    barras = ax.bar(df["comunidade"].astype(str), df["tamanho"])
    ax.set_title("Tamanho das comunidades na maior componente conexa")
    ax.set_xlabel("Comunidade")
    ax.set_ylabel("Quantidade de participantes")
    ax.grid(axis="y", alpha=0.25)

    for barra, valor in zip(barras, df["tamanho"]):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height(),
            str(int(valor)),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _salvar_figura(output_path)


def _filtrar_arestas_mais_fortes_por_no(G, top_n: int = 3):
    filtrado = nx.Graph()
    filtrado.add_nodes_from(G.nodes(data=True))

    arestas_selecionadas = set()
    for no in G.nodes:
        incidentes = sorted(
            G.edges(no, data=True),
            key=lambda item: float(item[2].get("weight", 0.0)),
            reverse=True,
        )
        for origem, destino, _ in incidentes[:top_n]:
            arestas_selecionadas.add(tuple(sorted((origem, destino))))

    for origem, destino in arestas_selecionadas:
        filtrado.add_edge(origem, destino, **G[origem][destino])
    return filtrado


def plotar_grafo_maior_componente_filtrado(
    G,
    output_path: str,
    top_labels: int = 8,
    top_edges_per_node: int = 3,
):
    """Gera grafo visual da maior componente com arestas mais fortes."""
    maior = extrair_maior_componente(G)
    subgrafo = _filtrar_arestas_mais_fortes_por_no(maior, top_edges_per_node)

    plt.figure(figsize=(12, 9))
    pos = _layout_grafo(subgrafo)
    comunidades = [
        int(_valor_no(subgrafo, no, "comunidade", 0))
        if "comunidade" in subgrafo.nodes[no]
        else 0
        for no in subgrafo.nodes
    ]

    nx.draw_networkx_edges(
        subgrafo,
        pos,
        alpha=0.35,
        width=_larguras_arestas(subgrafo),
        edge_color="0.4",
    )
    nx.draw_networkx_nodes(
        subgrafo,
        pos,
        node_size=_tamanhos_nos(subgrafo, "pagerank"),
        node_color=comunidades,
        cmap="tab20",
        alpha=0.92,
        linewidths=0.4,
        edgecolors="white",
    )
    nx.draw_networkx_labels(
        subgrafo,
        pos,
        labels=_ranking_nos_para_labels(subgrafo, top_labels),
        font_size=8,
        font_color="black",
    )

    plt.title("Maior componente conexa - conexões mais fortes (filtro visual)")
    plt.axis("off")
    _salvar_figura(output_path)


def _carregar_csv(caminho: Path, **kwargs) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo obrigatorio nao encontrado: {caminho}")
    return pd.read_csv(caminho, **kwargs)


def _buscar_sensibilidade(output_dir: Path) -> Path | None:
    candidatos = [
        output_dir / "tabelas" / "analise_sensibilidade_limiar.csv",
        Path("outputs") / "tabelas" / "analise_sensibilidade_limiar.csv",
    ]
    for caminho in candidatos:
        if caminho.exists():
            return caminho
    return None


def _buscar_palpites_processados() -> Path | None:
    caminho = Path("data") / "processed" / "palpites.csv"
    if caminho.exists():
        return caminho
    return None


def gerar_visualizacoes(
    output_dir: str,
    theta: float = 0.20,
) -> dict:
    """
    Gera as visualizacoes principais a partir das tabelas de saida do pipeline.

    Args:
        output_dir (str): Diretorio contendo `tabelas/`.
        theta (float): Limiar usado para reconstruir o grafo.

    Returns:
        dict: Caminhos das figuras geradas e avisos.
    """
    _validar_theta(theta)
    base = Path(output_dir)
    tabelas_dir = base / "tabelas"
    figuras_dir = base / "figuras"
    figuras_dir.mkdir(parents=True, exist_ok=True)

    pares_df = _carregar_csv(tabelas_dir / "pares_similaridade_final.csv")
    metricas_df = _carregar_csv(tabelas_dir / "metricas_participantes_final.csv")
    comunidades_df = _carregar_csv(tabelas_dir / "comunidades_final.csv")
    componentes_df = _carregar_csv(tabelas_dir / "componentes_final.csv")
    ranking_df = _carregar_csv(tabelas_dir / "ranking_centralidade.csv")
    matriz_df = _carregar_csv(tabelas_dir / "matriz_similaridade_final.csv", index_col=0)

    G = reconstruir_grafo_de_pares(
        pares_df,
        metricas_df=metricas_df,
        comunidades_df=comunidades_df,
        theta=theta,
    )

    ranking_participantes = ranking_df["participante_id"].tolist()
    caminhos = {
        "grafo_final_similaridade": str(figuras_dir / "grafo_final_similaridade.png"),
        "grafo_final_comunidades": str(figuras_dir / "grafo_final_comunidades.png"),
        "heatmap_similaridade_final": str(figuras_dir / "heatmap_similaridade_final.png"),
        "ranking_centralidade_pagerank": str(figuras_dir / "ranking_centralidade_pagerank.png"),
        "distribuicao_tamanho_componentes": str(
            figuras_dir / "distribuicao_tamanho_componentes.png"
        ),
        "grafo_maior_componente_comunidades": str(
            figuras_dir / "grafo_maior_componente_comunidades.png"
        ),
        "heatmap_similaridade_ordenado": str(figuras_dir / "heatmap_similaridade_ordenado.png"),
        "ranking_centralidade_pagerank_melhorado": str(
            figuras_dir / "ranking_centralidade_pagerank_melhorado.png"
        ),
        "distribuicao_tamanho_componentes_melhorado": str(
            figuras_dir / "distribuicao_tamanho_componentes_melhorado.png"
        ),
        "tamanho_comunidades_maior_componente": str(
            figuras_dir / "tamanho_comunidades_maior_componente.png"
        ),
        "grafo_maior_componente_filtrado": str(
            figuras_dir / "grafo_maior_componente_filtrado.png"
        ),
    }
    avisos: list[str] = []
    tabelas_artigo: dict[str, str] = {}

    plotar_grafo_final_similaridade(G, caminhos["grafo_final_similaridade"])
    plotar_grafo_final_comunidades(G, caminhos["grafo_final_comunidades"])
    plotar_heatmap_similaridade(
        matriz_df,
        caminhos["heatmap_similaridade_final"],
        participantes_ordenados=ranking_participantes,
    )
    plotar_ranking_centralidade(ranking_df, caminhos["ranking_centralidade_pagerank"])
    plotar_distribuicao_tamanho_componentes(
        componentes_df,
        caminhos["distribuicao_tamanho_componentes"],
    )
    plotar_grafo_maior_componente_comunidades(
        G,
        caminhos["grafo_maior_componente_comunidades"],
    )
    plotar_heatmap_similaridade_ordenado(
        matriz_df,
        metricas_df,
        caminhos["heatmap_similaridade_ordenado"],
    )
    plotar_ranking_centralidade_melhorado(
        ranking_df,
        caminhos["ranking_centralidade_pagerank_melhorado"],
    )

    resumo_componentes_df = gerar_resumo_componentes_artigo(componentes_df, str(base))
    tabelas_artigo["resumo_componentes_artigo"] = str(
        tabelas_dir / "resumo_componentes_artigo.csv"
    )
    plotar_distribuicao_tamanho_componentes_melhorado(
        resumo_componentes_df,
        caminhos["distribuicao_tamanho_componentes_melhorado"],
    )

    resumo_comunidades_df = gerar_resumo_comunidades_maior_componente(G, str(base))
    tabelas_artigo["resumo_comunidades_maior_componente"] = str(
        tabelas_dir / "resumo_comunidades_maior_componente.csv"
    )
    plotar_tamanho_comunidades_maior_componente(
        resumo_comunidades_df,
        caminhos["tamanho_comunidades_maior_componente"],
    )
    plotar_grafo_maior_componente_filtrado(
        G,
        caminhos["grafo_maior_componente_filtrado"],
    )

    palpites_path = _buscar_palpites_processados()
    if palpites_path is None:
        avisos.append(
            "Arquivos de data/processed incompletos; resumo de dados e distribuicao de palpites pulados."
        )
    else:
        caminhos["distribuicao_palpites_participante"] = str(
            figuras_dir / "distribuicao_palpites_participante.png"
        )
        try:
            gerar_resumo_dados_artigo("data/processed", str(base))
            tabelas_artigo["resumo_dados_artigo"] = str(tabelas_dir / "resumo_dados_artigo.csv")
        except FileNotFoundError:
            avisos.append(
                "Arquivos de data/processed incompletos; resumo_dados_artigo nao foi gerado."
            )
        palpites_df = pd.read_csv(palpites_path)
        plotar_distribuicao_palpites_participante(
            palpites_df,
            caminhos["distribuicao_palpites_participante"],
        )

    sensibilidade_path = _buscar_sensibilidade(base)
    if sensibilidade_path is None:
        avisos.append("Tabela de analise de sensibilidade do limiar nao encontrada.")
    else:
        sensibilidade_df = pd.read_csv(sensibilidade_path)
        caminhos["analise_sensibilidade_limiar"] = str(
            figuras_dir / "analise_sensibilidade_limiar.png"
        )
        plotar_analise_sensibilidade_limiar(
            sensibilidade_df,
            caminhos["analise_sensibilidade_limiar"],
        )

    return {
        "figuras": caminhos,
        "tabelas": tabelas_artigo,
        "avisos": avisos,
        "vertices": G.number_of_nodes(),
        "arestas": G.number_of_edges(),
    }
