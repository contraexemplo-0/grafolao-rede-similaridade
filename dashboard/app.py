"""Dashboard local em Streamlit para o módulo Laços de Palpite."""

from __future__ import annotations

import math
from html import escape
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st


USUARIO_ATUAL = "P_TESTE"
ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA_DIR = ROOT / "dashboard" / "data" / "demo"
OUTPUTS = {
    "Cobertura linear": ROOT / "outputs" / "dashboard_demo" / "linear",
    "Cobertura suavizada": ROOT / "outputs" / "dashboard_demo" / "suavizado",
}
THETA_DEMO = 0.20
MAX_ARESTAS_INTERATIVAS = 260
COMMUNITY_COLORS = [
    "#4ade80",
    "#22c55e",
    "#86efac",
    "#38bdf8",
    "#a78bfa",
    "#facc15",
    "#fb923c",
    "#f472b6",
    "#2dd4bf",
    "#c4b5fd",
]
PAGINAS = [
    "Visão geral",
    "Meus semelhantes",
    "Grupos de palpite",
    "Participantes centrais",
    "Mapa da rede",
]


st.set_page_config(
    page_title="Grafolão - Redes de Similaridade",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_css() -> None:
    css_path = ROOT / "dashboard" / "styles" / "app.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _read_csv(path: str, index_col=None) -> pd.DataFrame:
    caminho = Path(path)
    if not caminho.exists():
        return pd.DataFrame()
    return pd.read_csv(caminho, index_col=index_col)


def _table(output_dir: Path, name: str, index_col=None) -> pd.DataFrame:
    return _read_csv(str(output_dir / "tabelas" / name), index_col=index_col)


def _demo_csv(name: str) -> pd.DataFrame:
    return _read_csv(str(DEMO_DATA_DIR / name))


def _fmt(value, casas: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.{casas}f}"
    return str(value)


def _fmt_int(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        return f"{int(value):,}".replace(",", ".")
    except (TypeError, ValueError):
        return str(value)


def metric_card(title: str, value, subtitle: str | None = None) -> None:
    """Renderiza um card de métrica com a identidade visual do Grafolão."""
    subtitle_html = f'<div class="metric-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-title">{escape(title)}</div>
          <div class="metric-value">{escape(str(value))}</div>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_card_html(title: str, value, subtitle: str | None = None) -> str:
    subtitle_html = f'<div class="metric-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    return (
        '<div class="metric-card">'
        f'<div class="metric-title">{escape(title)}</div>'
        f'<div class="metric-value">{escape(str(value))}</div>'
        f"{subtitle_html}"
        "</div>"
    )


def metric_grid(cards: list[tuple[str, object, str | None]], columns: int) -> None:
    """Renderiza uma linha de cards em grid responsiva."""
    cards_html = "".join(_metric_card_html(title, value, subtitle) for title, value, subtitle in cards)
    st.markdown(
        f'<div class="metric-grid metric-grid-{columns}">{cards_html}</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str) -> None:
    """Renderiza cabeçalho de seção."""
    st.markdown(
        f"""
        <div class="section-header">
          <h2>{escape(title)}</h2>
          <p>{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str) -> None:
    """Renderiza bloco explicativo curto."""
    st.markdown(
        f"""
        <div class="info-card">
          <div class="info-card-title">{escape(title)}</div>
          <div class="info-card-body">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _fmt_percent(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        return f"{float(value) * 100:.1f}%".replace(".", ",")
    except (TypeError, ValueError):
        return "-"


def _fmt_decimal(value, casas: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        return f"{float(value):.{casas}f}".replace(".", ",")
    except (TypeError, ValueError):
        return "-"


def _fmt_rank(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        return f"{int(value)}º"
    except (TypeError, ValueError):
        return str(value)


def _nivel_similaridade(value) -> str:
    if value >= 0.70:
        return "Alta"
    if value >= 0.50:
        return "Média"
    return "Baixa"


def render_styled_table(rows: list[dict], columns: list[str], highlight_user: bool = True) -> None:
    """Renderiza tabela HTML escura com estilo do dashboard."""
    if not rows:
        st.info("Não há dados para exibir nesta tabela.")
        return

    header = "".join(f"<th>{escape(col)}</th>" for col in columns)
    body = []
    for row in rows:
        row_classes = []
        participante = str(row.get("Participante", ""))
        if highlight_user and participante == USUARIO_ATUAL:
            row_classes.append("current-user")
        posicao = str(row.get("Posição", ""))
        if posicao in {"1º", "2º", "3º"}:
            row_classes.append("top-rank")

        cells = []
        for col in columns:
            value = row.get(col, "-")
            if col == "Nível":
                level_class = str(value).lower().replace("é", "e")
                cells.append(f'<td><span class="level-badge level-{level_class}">{escape(str(value))}</span></td>')
            elif col == "Posição" and str(value) in {"1º", "2º", "3º"}:
                cells.append(f'<td><span class="rank-badge">{escape(str(value))}</span></td>')
            else:
                cells.append(f"<td>{escape(str(value))}</td>")
        body.append(f'<tr class="{" ".join(row_classes)}">{"".join(cells)}</tr>')

    st.markdown(
        f"""
        <div class="styled-table-wrap">
          <table class="styled-table">
            <thead><tr>{header}</tr></thead>
            <tbody>{''.join(body)}</tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _note(text: str) -> None:
    st.markdown(f'<div class="soft-note">{escape(text)}</div>', unsafe_allow_html=True)


def _missing_results(output_dir: Path) -> bool:
    required = [
        output_dir / "tabelas" / "resumo_grafo_final.csv",
        output_dir / "tabelas" / "pares_similaridade_final.csv",
        output_dir / "tabelas" / "ranking_centralidade.csv",
    ]
    return any(not path.exists() for path in required)


def _friendly_missing(output_dir: Path) -> None:
    st.warning(
        "Resultados do dashboard ainda não encontrados. Gere os dados demo e rode o pipeline antes de acessar esta página."
    )
    st.code(
        "python dashboard/gerar_dados_demo.py\n"
        "python dashboard/gerar_resultados_demo.py",
        language="bash",
    )
    st.caption(f"Pasta esperada: {output_dir}")


def _find_image(output_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        path = output_dir / "figuras" / name
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _show_image(path: Path | None, caption: str) -> None:
    if path is None:
        st.info("Figura ainda não disponível para este modelo.")
        return
    st.markdown('<div class="image-card">', unsafe_allow_html=True)
    st.image(str(path), use_container_width=True)
    st.markdown(f'<div class="caption">{escape(caption)}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _resumo_basico(output_dir: Path) -> dict:
    resumo = _table(output_dir, "resumo_grafo_final.csv")
    ranking = _table(output_dir, "ranking_centralidade.csv")
    comunidades = _table(output_dir, "comunidades_final.csv")
    participantes = _demo_csv("participantes.csv")
    jogos = _demo_csv("jogos.csv")
    palpites = _demo_csv("palpites.csv")

    dados = {}
    if not resumo.empty:
        dados.update(resumo.iloc[0].to_dict())
    dados["participantes"] = len(participantes)
    dados["jogos"] = len(jogos)
    dados["palpites"] = len(palpites)
    dados["participante_mais_central"] = ranking.iloc[0]["participante_id"] if not ranking.empty else "-"
    if not comunidades.empty and {"comunidade", "participante_id"}.issubset(comunidades.columns):
        tamanhos = comunidades.groupby("comunidade")["participante_id"].nunique()
        dados["comunidades_nao_triviais"] = int((tamanhos > 1).sum())
    else:
        dados["comunidades_nao_triviais"] = None
    return dados


def _participacao_palpites() -> dict[str, int]:
    palpites = _demo_csv("palpites.csv")
    if palpites.empty or "participante_id" not in palpites.columns:
        return {}
    return palpites.groupby("participante_id").size().astype(int).to_dict()


def _similaridades_com_p_teste(output_dir: Path) -> dict[str, float]:
    pares = _table(output_dir, "pares_similaridade_final.csv")
    similaridades = {USUARIO_ATUAL: 1.0}
    if pares.empty or not {"participante_u", "participante_v", "sim_final"}.issubset(pares.columns):
        return similaridades

    linhas = pares[
        (pares["participante_u"] == USUARIO_ATUAL) | (pares["participante_v"] == USUARIO_ATUAL)
    ]
    for _, row in linhas.iterrows():
        outro = row["participante_v"] if row["participante_u"] == USUARIO_ATUAL else row["participante_u"]
        if not pd.isna(row["sim_final"]):
            similaridades[str(outro)] = float(row["sim_final"])
    return similaridades


def _mapa_atributos_vertices(output_dir: Path) -> dict[str, dict]:
    metricas = _table(output_dir, "metricas_participantes_final.csv")
    ranking = _table(output_dir, "ranking_centralidade.csv")
    comunidades = _table(output_dir, "comunidades_final.csv")
    componentes = _table(output_dir, "componentes_final.csv")
    palpites_por_participante = _participacao_palpites()
    similaridades_p_teste = _similaridades_com_p_teste(output_dir)

    atributos: dict[str, dict] = {}
    for df in [metricas, ranking, comunidades, componentes]:
        if df.empty or "participante_id" not in df.columns:
            continue
        for _, row in df.iterrows():
            participante = str(row["participante_id"])
            atributos.setdefault(participante, {})
            for coluna, valor in row.items():
                if coluna != "participante_id" and not pd.isna(valor):
                    atributos[participante][coluna] = valor

    for participante, qtd in palpites_por_participante.items():
        atributos.setdefault(str(participante), {})["palpites"] = qtd

    for participante, similaridade in similaridades_p_teste.items():
        atributos.setdefault(str(participante), {})["similaridade_p_teste"] = similaridade

    return atributos


def _construir_grafo_interativo(output_dir: Path, sim_minima: float) -> nx.Graph:
    pares = _table(output_dir, "pares_similaridade_final.csv")
    atributos = _mapa_atributos_vertices(output_dir)
    G = nx.Graph()

    for participante, dados in atributos.items():
        G.add_node(participante, **dados)

    if pares.empty:
        return G

    colunas = {"participante_u", "participante_v", "sim_final"}
    if not colunas.issubset(pares.columns):
        return G

    pares_validos = pares[pares["sim_final"] >= sim_minima].copy()
    for _, row in pares_validos.iterrows():
        u = str(row["participante_u"])
        v = str(row["participante_v"])
        peso = float(row["sim_final"])
        for participante in [u, v]:
            if participante not in G:
                G.add_node(participante, **atributos.get(participante, {}))
        G.add_edge(u, v, weight=peso, distance=1 - peso)

    return G


def _arestas_p_teste(output_dir: Path, top_n: int) -> list[tuple[str, str, float]]:
    pares = _table(output_dir, "pares_similaridade_final.csv")
    if pares.empty or not {"participante_u", "participante_v", "sim_final"}.issubset(pares.columns):
        return []
    linhas = pares[
        (pares["participante_u"] == USUARIO_ATUAL) | (pares["participante_v"] == USUARIO_ATUAL)
    ].copy()
    if linhas.empty:
        return []
    linhas["outro"] = linhas.apply(
        lambda row: row["participante_v"] if row["participante_u"] == USUARIO_ATUAL else row["participante_u"],
        axis=1,
    )
    linhas = linhas.sort_values("sim_final", ascending=False).head(top_n)
    return [(USUARIO_ATUAL, str(row["outro"]), float(row["sim_final"])) for _, row in linhas.iterrows()]


def _filtrar_minha_rede(output_dir: Path, top_n: int) -> nx.Graph:
    atributos = _mapa_atributos_vertices(output_dir)
    G = nx.Graph()
    G.add_node(USUARIO_ATUAL, **atributos.get(USUARIO_ATUAL, {}))
    for u, v, peso in _arestas_p_teste(output_dir, top_n):
        for participante in [u, v]:
            if participante not in G:
                G.add_node(participante, **atributos.get(participante, {}))
        G.add_edge(u, v, weight=peso, distance=1 - peso)
    return G


def _filtrar_maior_componente(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G.copy()
    componentes = list(nx.connected_components(G))
    if not componentes:
        return G.copy()
    maior = max(componentes, key=lambda comp: (len(comp), USUARIO_ATUAL in comp))
    return G.subgraph(maior).copy()


def _filtrar_global_simplificada(G: nx.Graph, limite_nos: int = 70) -> nx.Graph:
    if G.number_of_nodes() <= limite_nos:
        H = G.copy()
    else:
        ordenados = sorted(
            G.nodes,
            key=lambda n: (
                n == USUARIO_ATUAL,
                float(G.nodes[n].get("pagerank", 0) or 0),
                int(G.degree(n)),
            ),
            reverse=True,
        )
        H = G.subgraph(ordenados[:limite_nos]).copy()

    if H.number_of_edges() > MAX_ARESTAS_INTERATIVAS:
        arestas = sorted(H.edges(data=True), key=lambda item: item[2].get("weight", 0), reverse=True)
        K = nx.Graph()
        for node, dados in H.nodes(data=True):
            K.add_node(node, **dados)
        for u, v, dados in arestas[:MAX_ARESTAS_INTERATIVAS]:
            K.add_edge(u, v, **dados)
        return K
    return H


def _layout_minha_rede(G: nx.Graph) -> dict[str, tuple[float, float]]:
    if USUARIO_ATUAL not in G:
        return nx.spring_layout(G, seed=42, weight="weight")
    pos = {USUARIO_ATUAL: (0.0, 0.0)}
    vizinhos = sorted(G.neighbors(USUARIO_ATUAL), key=lambda n: G[USUARIO_ATUAL][n].get("weight", 0), reverse=True)
    total = max(len(vizinhos), 1)
    for idx, node in enumerate(vizinhos):
        angulo = 2 * math.pi * idx / total
        raio = 1.0 + 0.25 * (idx // 10)
        pos[node] = (raio * math.cos(angulo), raio * math.sin(angulo))
    return pos


def _cor_comunidade(comunidade) -> str:
    if comunidade is None or pd.isna(comunidade):
        return "#94a3b8"
    try:
        idx = int(float(comunidade))
    except (TypeError, ValueError):
        idx = abs(hash(str(comunidade)))
    return COMMUNITY_COLORS[idx % len(COMMUNITY_COLORS)]


def _tamanho_no(dados: dict) -> float:
    pagerank = float(dados.get("pagerank", 0) or 0)
    grau = float(dados.get("grau", 0) or 0)
    if pagerank > 0:
        return 18 + min(42, pagerank * 900)
    return 16 + min(34, grau * 1.5)


def _hover_no(node: str, dados: dict) -> str:
    sim_p_teste = dados.get("similaridade_p_teste")
    similaridade_texto = _fmt_percent(sim_p_teste) if sim_p_teste is not None else "não disponível"
    return (
        f"<b>{node}</b><br>"
        f"Comunidade: {_fmt(dados.get('comunidade'))}<br>"
        f"PageRank: {_fmt(float(dados.get('pagerank', 0) or 0))}<br>"
        f"Grau: {_fmt(dados.get('grau'), 0)}<br>"
        f"Palpites: {_fmt(dados.get('palpites'), 0)}<br>"
        f"Similaridade com P_TESTE: {similaridade_texto}"
    )


def _criar_figura_rede(G: nx.Graph, modo: str, mostrar_rotulos: bool):
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        return None

    if G.number_of_nodes() == 0:
        return None

    if modo == "Minha rede":
        pos = _layout_minha_rede(G)
    else:
        pos = nx.spring_layout(G, seed=42, weight="weight", k=0.9)

    fig = go.Figure()

    for u, v, dados in G.edges(data=True):
        peso = float(dados.get("weight", 0) or 0)
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        largura = 0.8 + min(4.2, peso * 5)
        opacidade = 0.18 + min(0.46, peso * 0.72)
        cor = f"rgba(216, 255, 224, {opacidade:.2f})"
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"width": largura, "color": cor},
                hoverinfo="text",
                text=f"{u} ↔ {v}<br>Similaridade: {peso:.4f}",
                showlegend=False,
            )
        )

    nodes = list(G.nodes)
    top_label_nodes = {
        node
        for node, _ in sorted(
            G.nodes(data=True),
            key=lambda item: float(item[1].get("pagerank", 0) or 0),
            reverse=True,
        )[:8]
    }
    label_nodes = {USUARIO_ATUAL} | top_label_nodes if mostrar_rotulos else {USUARIO_ATUAL}

    fig.add_trace(
        go.Scatter(
            x=[pos[node][0] for node in nodes],
            y=[pos[node][1] for node in nodes],
            mode="markers+text",
            text=[node if node in label_nodes else "" for node in nodes],
            textposition="top center",
            textfont={"size": 11, "color": "#ffffff"},
            marker={
                "size": [
                    34 if node == USUARIO_ATUAL else _tamanho_no(G.nodes[node])
                    for node in nodes
                ],
                "color": [
                    "#facc15" if node == USUARIO_ATUAL else _cor_comunidade(G.nodes[node].get("comunidade"))
                    for node in nodes
                ],
                "line": {
                    "width": [4 if node == USUARIO_ATUAL else 1.4 for node in nodes],
                    "color": ["#ffffff" if node == USUARIO_ATUAL else "rgba(255,255,255,0.62)" for node in nodes],
                },
                "opacity": 0.96,
            },
            hoverinfo="text",
            hovertext=[_hover_no(node, G.nodes[node]) for node in nodes],
            showlegend=False,
        )
    )

    fig.update_layout(
        height=640,
        margin={"l": 10, "r": 10, "t": 18, "b": 10},
        paper_bgcolor="#0b3f24",
        plot_bgcolor="#0b3f24",
        font={"color": "#ffffff"},
        hoverlabel={
            "bgcolor": "#124f2f",
            "bordercolor": "#4ade80",
            "font": {"color": "#ffffff"},
        },
        xaxis={"visible": False, "showgrid": False, "zeroline": False},
        yaxis={"visible": False, "showgrid": False, "zeroline": False},
        dragmode="pan",
    )
    return fig


def _participante_mais_semelhante(output_dir: Path) -> str:
    arestas = _arestas_p_teste(output_dir, 1)
    if not arestas:
        return "-"
    return arestas[0][1]


def _comunidade_p_teste(output_dir: Path) -> str:
    comunidades = _table(output_dir, "comunidades_final.csv")
    if comunidades.empty or "participante_id" not in comunidades.columns or "comunidade" not in comunidades.columns:
        return "-"
    linha = comunidades[comunidades["participante_id"] == USUARIO_ATUAL]
    if linha.empty:
        return "-"
    return _fmt(linha.iloc[0]["comunidade"], 0)


def render_sidebar() -> str:
    """Renderiza a sidebar com a navegação visual do Grafolão e retorna o modelo."""
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
              <div class="brand-mark">G</div>
              <div>
                <div class="brand-title">Grafolão</div>
                <div class="brand-subtitle">Copa 2026</div>
              </div>
            </div>
            <div class="sidebar-section">Principal</div>
            <div class="sidebar-item">▦ Painel</div>
            <div class="sidebar-item">▤ Jogos e Palpites</div>
            <div class="sidebar-item">▤ Meus Palpites</div>
            <div class="sidebar-item">♕ Ranking</div>
            <div class="sidebar-item">◎ Grupos</div>
            <div class="sidebar-item">? Ajuda</div>
            <div class="sidebar-section">Análise de Grafos</div>
            <div class="sidebar-item">⌁ Grafo de Confrontos</div>
            <div class="sidebar-item">⌘ Caminho Mínimo para o Título</div>
            <div class="sidebar-item">▱ Cliques e Panelinhas</div>
            <div class="sidebar-item">◇ Aventura da Zebra</div>
            <div class="sidebar-item active">⬡ Redes de Similaridade</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="block-label">Modelo de análise</div>', unsafe_allow_html=True)
        modelo = st.selectbox(
            "Modelo de análise",
            list(OUTPUTS.keys()),
            label_visibility="collapsed",
        )
        st.markdown(
            f"""
            <div class="sidebar-user">
              <strong>{USUARIO_ATUAL}</strong>
              <span>Usuário demo fixo</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return modelo


def render_topbar() -> None:
    st.markdown(
        """
        <div class="topbar">
          <div class="topbar-title">Grafolão da Copa: Redes de Similaridade</div>
          <div class="topbar-chip">Tema escuro</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
          <h1>Olá, {USUARIO_ATUAL} 👋</h1>
          <p>Veja quem palpita de forma parecida com você</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_module_nav() -> str:
    return st.radio(
        "Navegação do módulo",
        PAGINAS,
        horizontal=True,
        label_visibility="collapsed",
    )


def page_visao_geral(output_dir: Path, modelo: str) -> None:
    section_header(
        "Visão geral",
        "Resumo da rede de similaridade para o usuário demo e o modelo selecionado.",
    )
    if _missing_results(output_dir):
        _friendly_missing(output_dir)
        return

    dados = _resumo_basico(output_dir)
    metric_grid(
        [
            ("Participantes", _fmt_int(dados.get("participantes", 0)), "usuários anonimizados"),
            ("Arestas", _fmt_int(dados.get("arestas", 0)), "relações de similaridade"),
            ("Componentes", _fmt_int(dados.get("componentes", 0)), "grupos desconectados"),
            ("Maior componente", _fmt_int(dados.get("maior_componente", 0)), "maior núcleo conectado"),
            ("Mais central", dados.get("participante_mais_central", "-"), "PageRank"),
        ],
        columns=5,
    )

    metric_grid(
        [
            ("Usuário atual", USUARIO_ATUAL, "participante artificial"),
            (
                "Comunidades não triviais",
                _fmt_int(dados.get("comunidades_nao_triviais")),
                "grupos com mais de 1 participante",
            ),
            ("Palpites", _fmt_int(dados.get("palpites", 0)), f"{_fmt_int(dados.get('jogos', 0))} jogos"),
        ],
        columns=3,
    )

    info_card(
        "Como interpretar",
        "Este módulo não mede quem acertou mais. Ele mede similaridade entre padrões de palpite.",
    )


def page_meus_semelhantes(output_dir: Path) -> None:
    section_header(
        "Meus semelhantes",
        "Participantes com padrões de palpite mais próximos do seu.",
    )
    if _missing_results(output_dir):
        _friendly_missing(output_dir)
        return

    pares = _table(output_dir, "pares_similaridade_final.csv")
    if pares.empty or USUARIO_ATUAL not in set(pares["participante_u"]).union(set(pares["participante_v"])):
        st.info("P_TESTE ainda não apareceu nos pares de similaridade. Gere os dados demo e os resultados novamente.")
        return

    linhas = pares[
        (pares["participante_u"] == USUARIO_ATUAL) | (pares["participante_v"] == USUARIO_ATUAL)
    ].copy()
    linhas["participante"] = linhas.apply(
        lambda row: row["participante_v"] if row["participante_u"] == USUARIO_ATUAL else row["participante_u"],
        axis=1,
    )
    linhas = linhas.sort_values("sim_final", ascending=False).reset_index(drop=True)
    top = linhas.head(10)

    cols = st.columns(3)
    with cols[0]:
        metric_card("Participante mais parecido com você", top.iloc[0]["participante"] if not top.empty else "-")
    with cols[1]:
        metric_card("Maior similaridade", _fmt(float(top.iloc[0]["sim_final"])) if not top.empty else "0.0000")
    with cols[2]:
        metric_card("Conexões de P_TESTE", _fmt_int((linhas["sim_final"] >= 0.20).sum()))

    info_card(
        "Similaridade de palpites",
        "A similaridade considera resultado previsto, saldo, placar e jogos em comum.",
    )

    rows = []
    for _, row in top.iterrows():
        sim = float(row["sim_final"])
        rows.append(
            {
                "Participante": row["participante"],
                "Similaridade": _fmt_percent(sim),
                "Jogos em comum": _fmt_int(row["jogos_comparaveis"]) if "jogos_comparaveis" in row else "-",
                "Cobertura": _fmt_percent(row["cobertura"]) if "cobertura" in row else "-",
                "Nível": _nivel_similaridade(sim),
            }
        )
    render_styled_table(rows, ["Participante", "Similaridade", "Jogos em comum", "Cobertura", "Nível"])


def page_grupos(output_dir: Path) -> None:
    section_header(
        "Grupos de palpite",
        "Seu grupo reúne participantes com padrões de palpite semelhantes.",
    )
    if _missing_results(output_dir):
        _friendly_missing(output_dir)
        return

    comunidades = _table(output_dir, "comunidades_final.csv")
    resumo = _table(output_dir, "resumo_comunidades_maior_componente.csv")
    if comunidades.empty or USUARIO_ATUAL not in set(comunidades["participante_id"]):
        st.info("P_TESTE ainda não aparece em uma comunidade calculada.")
        return

    comunidade = comunidades.loc[comunidades["participante_id"] == USUARIO_ATUAL, "comunidade"].iloc[0]
    membros = comunidades[comunidades["comunidade"] == comunidade]["participante_id"].tolist()

    central = "-"
    if not resumo.empty and "comunidade" in resumo.columns:
        linha = resumo[resumo["comunidade"].astype(str) == str(comunidade)]
        if not linha.empty and "participante_mais_central" in linha.columns:
            central = linha.iloc[0]["participante_mais_central"]

    cols = st.columns(3)
    with cols[0]:
        metric_card("Comunidade de P_TESTE", comunidade)
    with cols[1]:
        metric_card("Tamanho da comunidade", _fmt_int(len(membros)))
    with cols[2]:
        metric_card("Participante mais central", central)

    info_card(
        "Seu grupo de palpite",
        "Comunidade significa agrupamento por semelhança de palpites. Não significa amizade, cópia ou torcida igual.",
    )
    metricas = _table(output_dir, "metricas_participantes_final.csv")
    pares = _table(output_dir, "pares_similaridade_final.csv")
    metricas_map = {}
    if not metricas.empty and "participante_id" in metricas.columns:
        metricas_map = metricas.set_index("participante_id").to_dict(orient="index")

    similaridade_map = {USUARIO_ATUAL: None}
    if not pares.empty and {"participante_u", "participante_v", "sim_final"}.issubset(pares.columns):
        linhas_teste = pares[
            (pares["participante_u"] == USUARIO_ATUAL) | (pares["participante_v"] == USUARIO_ATUAL)
        ].copy()
        for _, row in linhas_teste.iterrows():
            outro = row["participante_v"] if row["participante_u"] == USUARIO_ATUAL else row["participante_u"]
            similaridade_map[str(outro)] = float(row["sim_final"])

    rows = []
    for participante in membros:
        dados = metricas_map.get(participante, {})
        sim = similaridade_map.get(participante)
        rows.append(
            {
                "Participante": participante,
                "Comunidade": _fmt(comunidade, 0),
                "PageRank": _fmt_decimal(dados.get("pagerank"), 4),
                "Grau": _fmt_int(dados.get("grau", 0)),
                "Similaridade com P_TESTE": "Você" if participante == USUARIO_ATUAL else _fmt_percent(sim),
                "_pagerank_ordem": float(dados.get("pagerank", 0) or 0),
            }
        )
    rows = sorted(rows, key=lambda item: (item["Participante"] != USUARIO_ATUAL, -item["_pagerank_ordem"]))
    for row in rows:
        row.pop("_pagerank_ordem", None)
    render_styled_table(
        rows,
        ["Participante", "Comunidade", "PageRank", "Grau", "Similaridade com P_TESTE"],
    )

    imagem = _find_image(
        output_dir,
        [
            "grafo_maior_componente_comunidades_organizado.png",
            "grafo_maior_componente_comunidades.png",
        ],
    )
    _show_image(imagem, "Maior componente da rede colorida por comunidades.")


def page_centrais(output_dir: Path) -> None:
    section_header(
        "Participantes centrais",
        "Participantes centrais são aqueles mais bem posicionados na rede de similaridade.",
    )
    if _missing_results(output_dir):
        _friendly_missing(output_dir)
        return

    ranking = _table(output_dir, "ranking_centralidade.csv")
    if ranking.empty:
        st.info("Ranking ainda não disponível.")
        return

    posicao = "-"
    linha_teste = ranking[ranking["participante_id"] == USUARIO_ATUAL]
    if not linha_teste.empty:
        posicao = int(linha_teste.iloc[0]["posicao"])

    cols = st.columns(3)
    with cols[0]:
        metric_card("Posição de P_TESTE", posicao)
    with cols[1]:
        metric_card("Líder do ranking", ranking.iloc[0]["participante_id"])
    with cols[2]:
        metric_card("Participantes exibidos", "Top 10")

    info_card(
        "Centralidade estrutural",
        "Centralidade não é pontuação. É posição estrutural na rede de similaridade.",
    )
    rows = []
    for _, row in ranking.head(10).iterrows():
        rows.append(
            {
                "Posição": _fmt_rank(row.get("posicao")),
                "Participante": row.get("participante_id", "-"),
                "Centralidade": _fmt_decimal(row.get("pagerank"), 4),
                "Grau": _fmt_int(row.get("grau", 0)),
                "Força": _fmt_decimal(row.get("forca"), 2),
                "Comunidade": _fmt(row.get("comunidade"), 0),
            }
        )
    render_styled_table(rows, ["Posição", "Participante", "Centralidade", "Grau", "Força", "Comunidade"])

    imagem = _find_image(output_dir, ["ranking_centralidade_pagerank_melhorado.png"])
    _show_image(imagem, "Top participantes por centralidade PageRank.")


def page_mapa(output_dir: Path) -> None:
    section_header(
        "Mapa da rede",
        "Explore as relações de similaridade entre participantes do bolão.",
    )
    if _missing_results(output_dir):
        _friendly_missing(output_dir)
        return

    modo_col, viz_col, sim_col, label_col = st.columns([1.3, 1, 1, 0.9])
    with modo_col:
        modo = st.selectbox(
            "Modo de visualização",
            ["Minha rede", "Maior componente", "Rede global simplificada"],
        )
    with viz_col:
        top_n = st.selectbox("Vizinhos de P_TESTE", [10, 15, 20, 30], index=1)
    with sim_col:
        sim_minima = st.slider("Similaridade mínima", 0.0, 0.8, THETA_DEMO, 0.05)
    with label_col:
        mostrar_rotulos = st.checkbox("Exibir rótulos", value=False)

    if modo == "Minha rede":
        G = _filtrar_minha_rede(output_dir, int(top_n))
    else:
        G_base = _construir_grafo_interativo(output_dir, float(sim_minima))
        if modo == "Maior componente":
            G = _filtrar_maior_componente(G_base)
        else:
            G = _filtrar_global_simplificada(G_base)

    if G.number_of_nodes() == 0:
        st.warning("A rede ainda não foi gerada para este modelo ou não há arestas no filtro selecionado.")
        return

    if USUARIO_ATUAL not in G:
        st.info("P_TESTE não aparece no recorte visual atual. Reduza o filtro de similaridade ou use o modo Minha rede.")

    metric_grid(
        [
            ("Usuário atual", USUARIO_ATUAL, "participante destacado"),
            ("Nós exibidos", _fmt_int(G.number_of_nodes()), "participantes no mapa"),
            ("Arestas exibidas", _fmt_int(G.number_of_edges()), "relações de similaridade"),
            ("Comunidade de P_TESTE", _comunidade_p_teste(output_dir), "quando disponível"),
            ("Mais semelhante", _participante_mais_semelhante(output_dir), "maior sim_final"),
        ],
        columns=5,
    )

    info_card(
        "Como ler o mapa",
        "Nós próximos ou conectados indicam padrões de palpite semelhantes. A posição visual ajuda na interpretação, mas a similaridade real está nas arestas e seus pesos.",
    )

    fig = _criar_figura_rede(G, modo, mostrar_rotulos)
    if fig is None:
        st.warning(
            "A visualização interativa usa Plotly. Instale as dependências do dashboard para ativar o mapa interativo."
        )
        st.code("pip install -r dashboard/requirements.txt", language="bash")
        imagem = _find_image(
            output_dir,
            [
                "grafo_maior_componente_comunidades_organizado.png",
                "grafo_maior_componente_comunidades.png",
                "grafo_completo_comunidades.png",
                "grafo_final_similaridade.png",
            ],
        )
        _show_image(imagem, "Fallback estático: nó = participante; cor = comunidade; aresta = similaridade.")
        return

    st.markdown('<div class="network-card">', unsafe_allow_html=True)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    st.markdown(
        """
        <div class="network-legend">
          <span><strong>Nó</strong>: participante</span>
          <span><strong>Cor</strong>: comunidade</span>
          <span><strong>Tamanho</strong>: centralidade</span>
          <span><strong>Linha</strong>: similaridade</span>
          <span><strong>Destaque</strong>: P_TESTE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    _load_css()
    modelo = render_sidebar()
    render_topbar()
    render_hero()

    pagina = render_module_nav()
    output_dir = OUTPUTS[modelo]

    if pagina == "Visão geral":
        page_visao_geral(output_dir, modelo)
    elif pagina == "Meus semelhantes":
        page_meus_semelhantes(output_dir)
    elif pagina == "Grupos de palpite":
        page_grupos(output_dir)
    elif pagina == "Participantes centrais":
        page_centrais(output_dir)
    else:
        page_mapa(output_dir)


if __name__ == "__main__":
    main()
