from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.visualizacao import (
    extrair_maior_componente,
    gerar_resumo_componentes_artigo,
    gerar_resumo_comunidades_maior_componente,
    gerar_resumo_dados_artigo,
    gerar_visualizacoes,
    plotar_analise_sensibilidade_limiar,
    plotar_distribuicao_palpites_participante,
    plotar_distribuicao_tamanho_componentes,
    plotar_distribuicao_tamanho_componentes_melhorado,
    plotar_grafo_final_comunidades,
    plotar_grafo_final_similaridade,
    plotar_grafo_maior_componente_comunidades,
    plotar_grafo_maior_componente_filtrado,
    plotar_heatmap_similaridade,
    plotar_heatmap_similaridade_ordenado,
    plotar_ranking_centralidade,
    plotar_ranking_centralidade_melhorado,
    plotar_tamanho_comunidades_maior_componente,
    reconstruir_grafo_de_pares,
)


def _pares_df():
    return pd.DataFrame(
        [
            {"participante_u": "P001", "participante_v": "P002", "sim_final": 0.8},
            {"participante_u": "P001", "participante_v": "P003", "sim_final": 0.4},
            {"participante_u": "P002", "participante_v": "P003", "sim_final": 0.6},
        ]
    )


def _metricas_df():
    return pd.DataFrame(
        [
            {
                "participante_id": "P001",
                "grau": 2,
                "forca": 1.2,
                "betweenness": 0.0,
                "pagerank": 0.4,
                "componente": 1,
                "comunidade": 1,
            },
            {
                "participante_id": "P002",
                "grau": 2,
                "forca": 1.4,
                "betweenness": 0.5,
                "pagerank": 0.35,
                "componente": 1,
                "comunidade": 1,
            },
            {
                "participante_id": "P003",
                "grau": 2,
                "forca": 1.0,
                "betweenness": 0.0,
                "pagerank": 0.25,
                "componente": 1,
                "comunidade": 2,
            },
        ]
    )


def _comunidades_df():
    return pd.DataFrame(
        [
            {"comunidade": 1, "participante_id": "P001"},
            {"comunidade": 1, "participante_id": "P002"},
            {"comunidade": 2, "participante_id": "P003"},
        ]
    )


def _componentes_df():
    return pd.DataFrame(
        [
            {"componente": 1, "participante_id": "P001"},
            {"componente": 1, "participante_id": "P002"},
            {"componente": 1, "participante_id": "P003"},
        ]
    )


def _palpites_df():
    return pd.DataFrame(
        [
            {"participante_id": "P001", "jogo_id": "J001"},
            {"participante_id": "P001", "jogo_id": "J002"},
            {"participante_id": "P002", "jogo_id": "J001"},
            {"participante_id": "P003", "jogo_id": "J001"},
            {"participante_id": "P003", "jogo_id": "J002"},
            {"participante_id": "P003", "jogo_id": "J003"},
        ]
    )


def _ranking_df():
    return pd.DataFrame(
        [
            {
                "posicao": 1,
                "participante_id": "P001",
                "pagerank": 0.4,
                "grau": 2,
                "forca": 1.2,
                "betweenness": 0.0,
                "comunidade": 1,
            },
            {
                "posicao": 2,
                "participante_id": "P002",
                "pagerank": 0.35,
                "grau": 2,
                "forca": 1.4,
                "betweenness": 0.5,
                "comunidade": 1,
            },
            {
                "posicao": 3,
                "participante_id": "P003",
                "pagerank": 0.25,
                "grau": 2,
                "forca": 1.0,
                "betweenness": 0.0,
                "comunidade": 2,
            },
        ]
    )


def _matriz_df():
    participantes = ["P001", "P002", "P003"]
    return pd.DataFrame(
        [
            [1.0, 0.8, 0.4],
            [0.8, 1.0, 0.6],
            [0.4, 0.6, 1.0],
        ],
        index=participantes,
        columns=participantes,
    )


def _sensibilidade_df():
    return pd.DataFrame(
        [
            {"theta": 0.2, "arestas": 3, "componentes": 1, "maior_componente": 3},
            {"theta": 0.5, "arestas": 2, "componentes": 1, "maior_componente": 3},
            {"theta": 0.7, "arestas": 1, "componentes": 2, "maior_componente": 2},
        ]
    )


def _assert_png(caminho: Path):
    assert caminho.exists()
    assert caminho.stat().st_size > 0


def test_reconstruir_grafo_de_pares_cria_nos_arestas_e_respeita_theta():
    G = reconstruir_grafo_de_pares(
        _pares_df(),
        metricas_df=_metricas_df(),
        comunidades_df=_comunidades_df(),
        theta=0.5,
    )

    assert set(G.nodes) == {"P001", "P002", "P003"}
    assert G.has_edge("P001", "P002")
    assert G.has_edge("P002", "P003")
    assert not G.has_edge("P001", "P003")
    assert G["P001"]["P002"]["weight"] == pytest.approx(0.8)
    assert G["P001"]["P002"]["distance"] == pytest.approx(0.2)


def test_reconstruir_grafo_de_pares_adiciona_atributos_vertices():
    G = reconstruir_grafo_de_pares(_pares_df(), metricas_df=_metricas_df(), theta=0.5)

    assert G.nodes["P001"]["pagerank"] == pytest.approx(0.4)
    assert G.nodes["P002"]["grau"] == 2
    assert G.nodes["P003"]["comunidade"] == 2


def test_extrair_maior_componente_retorna_apenas_maior_componente():
    pares = pd.concat(
        [
            _pares_df(),
            pd.DataFrame(
                [
                    {
                        "participante_u": "P004",
                        "participante_v": "P005",
                        "sim_final": 0.9,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    metricas = pd.concat(
        [
            _metricas_df(),
            pd.DataFrame(
                [
                    {"participante_id": "P004", "pagerank": 0.1, "comunidade": 3},
                    {"participante_id": "P005", "pagerank": 0.1, "comunidade": 3},
                    {"participante_id": "P006", "pagerank": 0.0, "comunidade": 4},
                ]
            ),
        ],
        ignore_index=True,
    )

    G = reconstruir_grafo_de_pares(pares, metricas_df=metricas, theta=0.5)
    maior = extrair_maior_componente(G)

    assert set(maior.nodes) == {"P001", "P002", "P003"}


@pytest.mark.parametrize("theta", [-0.1, 1.1])
def test_reconstruir_grafo_de_pares_rejeita_theta_invalido(theta):
    with pytest.raises(ValueError):
        reconstruir_grafo_de_pares(_pares_df(), theta=theta)


def test_funcoes_de_plot_geram_pngs(tmp_path):
    G = reconstruir_grafo_de_pares(
        _pares_df(),
        metricas_df=_metricas_df(),
        comunidades_df=_comunidades_df(),
        theta=0.5,
    )

    caminhos = {
        "grafo": tmp_path / "grafo.png",
        "comunidades": tmp_path / "comunidades.png",
        "maior_componente": tmp_path / "maior_componente.png",
        "maior_componente_filtrado": tmp_path / "maior_componente_filtrado.png",
        "heatmap": tmp_path / "heatmap.png",
        "heatmap_ordenado": tmp_path / "heatmap_ordenado.png",
        "ranking": tmp_path / "ranking.png",
        "ranking_melhorado": tmp_path / "ranking_melhorado.png",
        "sensibilidade": tmp_path / "sensibilidade.png",
        "palpites": tmp_path / "palpites.png",
        "componentes": tmp_path / "componentes.png",
        "componentes_melhorado": tmp_path / "componentes_melhorado.png",
        "comunidades_maior": tmp_path / "comunidades_maior.png",
    }

    plotar_grafo_final_similaridade(G, str(caminhos["grafo"]))
    plotar_grafo_final_comunidades(G, str(caminhos["comunidades"]))
    plotar_grafo_maior_componente_comunidades(G, str(caminhos["maior_componente"]))
    plotar_grafo_maior_componente_filtrado(G, str(caminhos["maior_componente_filtrado"]))
    plotar_heatmap_similaridade(_matriz_df(), str(caminhos["heatmap"]))
    plotar_heatmap_similaridade_ordenado(
        _matriz_df(),
        _metricas_df(),
        str(caminhos["heatmap_ordenado"]),
    )
    plotar_ranking_centralidade(_ranking_df(), str(caminhos["ranking"]))
    plotar_ranking_centralidade_melhorado(_ranking_df(), str(caminhos["ranking_melhorado"]))
    plotar_analise_sensibilidade_limiar(_sensibilidade_df(), str(caminhos["sensibilidade"]))
    plotar_distribuicao_palpites_participante(_palpites_df(), str(caminhos["palpites"]))
    plotar_distribuicao_tamanho_componentes(_componentes_df(), str(caminhos["componentes"]))
    resumo_componentes = gerar_resumo_componentes_artigo(_componentes_df(), str(tmp_path))
    resumo_comunidades = gerar_resumo_comunidades_maior_componente(G, str(tmp_path))
    plotar_distribuicao_tamanho_componentes_melhorado(
        resumo_componentes,
        str(caminhos["componentes_melhorado"]),
    )
    plotar_tamanho_comunidades_maior_componente(
        resumo_comunidades,
        str(caminhos["comunidades_maior"]),
    )

    for caminho in caminhos.values():
        _assert_png(caminho)


def test_gerar_tabelas_artigo_com_colunas_obrigatorias(tmp_path):
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    output_dir = tmp_path / "saida"
    output_dir.mkdir()
    G = reconstruir_grafo_de_pares(
        _pares_df(),
        metricas_df=_metricas_df(),
        comunidades_df=_comunidades_df(),
        theta=0.5,
    )

    pd.DataFrame(
        [
            {"participante_id": "P001", "rotulo": "Participante 001"},
            {"participante_id": "P002", "rotulo": "Participante 002"},
            {"participante_id": "P003", "rotulo": "Participante 003"},
        ]
    ).to_csv(data_dir / "participantes.csv", index=False)
    pd.DataFrame(
        [
            {"jogo_id": "J001"},
            {"jogo_id": "J002"},
            {"jogo_id": "J003"},
        ]
    ).to_csv(data_dir / "jogos.csv", index=False)
    _palpites_df().to_csv(data_dir / "palpites.csv", index=False)
    pd.DataFrame(
        [
            {"jogo_id": "J001"},
            {"jogo_id": "J002"},
        ]
    ).to_csv(data_dir / "resultados.csv", index=False)

    resumo_dados = gerar_resumo_dados_artigo(str(data_dir), str(output_dir))
    resumo_componentes = gerar_resumo_componentes_artigo(_componentes_df(), str(output_dir))
    resumo_comunidades = gerar_resumo_comunidades_maior_componente(G, str(output_dir))

    assert {
        "participantes_validos",
        "jogos_validos",
        "palpites_validos",
        "resultados_validos",
        "media_palpites_por_participante",
        "mediana_palpites_por_participante",
        "min_palpites_por_participante",
        "max_palpites_por_participante",
        "participantes_com_1_palpite",
        "participantes_com_ate_4_palpites",
    }.issubset(resumo_dados.columns)
    assert {
        "tamanho_componente",
        "quantidade_componentes",
        "quantidade_vertices_total",
    }.issubset(resumo_componentes.columns)
    assert {
        "comunidade",
        "tamanho",
        "arestas_internas",
        "densidade_interna",
        "forca_media",
        "pagerank_total",
        "participante_mais_central",
        "pagerank_participante_mais_central",
    }.issubset(resumo_comunidades.columns)

    assert (output_dir / "tabelas" / "resumo_dados_artigo.csv").exists()
    assert (output_dir / "tabelas" / "resumo_componentes_artigo.csv").exists()
    assert (output_dir / "tabelas" / "resumo_comunidades_maior_componente.csv").exists()


def test_funcoes_nao_quebram_com_muitos_vertices_isolados(tmp_path):
    pares = pd.DataFrame(
        [{"participante_u": "P001", "participante_v": "P002", "sim_final": 0.8}]
    )
    metricas = pd.DataFrame(
        [
            {"participante_id": f"P{indice:03d}", "pagerank": 0.01, "comunidade": indice}
            for indice in range(1, 16)
        ]
    )
    G = reconstruir_grafo_de_pares(pares, metricas_df=metricas, theta=0.5)
    caminho = tmp_path / "maior_isolados.png"

    plotar_grafo_maior_componente_comunidades(G, str(caminho))

    _assert_png(caminho)
    assert set(extrair_maior_componente(G).nodes) == {"P001", "P002"}


def test_gerar_visualizacoes_cria_figuras_principais_com_tabelas_minimas(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "saida"
    tabelas_dir = output_dir / "tabelas"
    tabelas_dir.mkdir(parents=True)
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)

    _pares_df().to_csv(tabelas_dir / "pares_similaridade_final.csv", index=False)
    _metricas_df().to_csv(tabelas_dir / "metricas_participantes_final.csv", index=False)
    _comunidades_df().to_csv(tabelas_dir / "comunidades_final.csv", index=False)
    _componentes_df().to_csv(tabelas_dir / "componentes_final.csv", index=False)
    _ranking_df().to_csv(tabelas_dir / "ranking_centralidade.csv", index=False)
    _matriz_df().to_csv(tabelas_dir / "matriz_similaridade_final.csv")
    _sensibilidade_df().to_csv(tabelas_dir / "analise_sensibilidade_limiar.csv", index=False)
    pd.DataFrame(
        [
            {"participante_id": "P001", "rotulo": "Participante 001"},
            {"participante_id": "P002", "rotulo": "Participante 002"},
            {"participante_id": "P003", "rotulo": "Participante 003"},
        ]
    ).to_csv(data_processed / "participantes.csv", index=False)
    pd.DataFrame(
        [
            {"jogo_id": "J001"},
            {"jogo_id": "J002"},
            {"jogo_id": "J003"},
        ]
    ).to_csv(data_processed / "jogos.csv", index=False)
    _palpites_df().to_csv(data_processed / "palpites.csv", index=False)
    pd.DataFrame(
        [
            {"jogo_id": "J001"},
            {"jogo_id": "J002"},
        ]
    ).to_csv(data_processed / "resultados.csv", index=False)

    metricas_path = tabelas_dir / "metricas_participantes_final.csv"
    metricas_antes = metricas_path.read_text(encoding="utf-8")

    resultado = gerar_visualizacoes(str(output_dir), theta=0.5)

    assert (output_dir / "figuras").exists()
    assert not resultado["avisos"]
    esperadas = {
        "grafo_final_similaridade",
        "grafo_final_comunidades",
        "heatmap_similaridade_final",
        "ranking_centralidade_pagerank",
        "analise_sensibilidade_limiar",
        "distribuicao_palpites_participante",
        "distribuicao_tamanho_componentes",
        "grafo_maior_componente_comunidades",
        "heatmap_similaridade_ordenado",
        "ranking_centralidade_pagerank_melhorado",
        "distribuicao_tamanho_componentes_melhorado",
        "tamanho_comunidades_maior_componente",
        "grafo_maior_componente_filtrado",
    }
    assert set(resultado["figuras"]) == esperadas
    assert {
        "resumo_dados_artigo",
        "resumo_componentes_artigo",
        "resumo_comunidades_maior_componente",
    }.issubset(resultado["tabelas"])
    for caminho in resultado["figuras"].values():
        _assert_png(Path(caminho))
    assert metricas_path.read_text(encoding="utf-8") == metricas_antes
