import json
from pathlib import Path

import pandas as pd
import pytest

from src.exportacao import (
    exportar_grafo_graphml,
    exportar_grafo_json,
    gerar_relatorio_final,
    reconstruir_grafo_final,
)
from src.validacao import gerar_validacao_final


def _criar_saida_minima(base: Path) -> None:
    tabelas = base / "tabelas"
    figuras = base / "figuras"
    resumo = base / "resumo"
    grafos = base / "grafos"
    for diretorio in [tabelas, figuras, resumo, grafos]:
        diretorio.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"participante_u": "P001", "participante_v": "P002", "sim_final": 0.7},
            {"participante_u": "P001", "participante_v": "P003", "sim_final": 0.1},
            {"participante_u": "P002", "participante_v": "P003", "sim_final": 0.5},
        ]
    ).to_csv(tabelas / "pares_similaridade_final.csv", index=False)

    pd.DataFrame(
        [
            {
                "participante_id": "P001",
                "grau": 1,
                "forca": 0.7,
                "centralidade_grau": 0.5,
                "betweenness": 0.0,
                "agrupamento_local": 0.0,
                "pagerank": 0.4,
                "componente": 1,
                "comunidade": 1,
            },
            {
                "participante_id": "P002",
                "grau": 2,
                "forca": 1.2,
                "centralidade_grau": 1.0,
                "betweenness": 1.0,
                "agrupamento_local": 0.0,
                "pagerank": 0.35,
                "componente": 1,
                "comunidade": 1,
            },
            {
                "participante_id": "P003",
                "grau": 1,
                "forca": 0.5,
                "centralidade_grau": 0.5,
                "betweenness": 0.0,
                "agrupamento_local": 0.0,
                "pagerank": 0.25,
                "componente": 1,
                "comunidade": 2,
            },
        ]
    ).to_csv(tabelas / "metricas_participantes_final.csv", index=False)

    pd.DataFrame(
        [
            {"componente": 1, "participante_id": "P001"},
            {"componente": 1, "participante_id": "P002"},
            {"componente": 1, "participante_id": "P003"},
        ]
    ).to_csv(tabelas / "componentes_final.csv", index=False)

    pd.DataFrame(
        [
            {"comunidade": 1, "participante_id": "P001"},
            {"comunidade": 1, "participante_id": "P002"},
            {"comunidade": 2, "participante_id": "P003"},
        ]
    ).to_csv(tabelas / "comunidades_final.csv", index=False)

    pd.DataFrame(
        [
            {
                "posicao": 1,
                "participante_id": "P001",
                "pagerank": 0.4,
                "grau": 1,
                "forca": 0.7,
                "betweenness": 0.0,
                "comunidade": 1,
            },
            {
                "posicao": 2,
                "participante_id": "P002",
                "pagerank": 0.35,
                "grau": 2,
                "forca": 1.2,
                "betweenness": 1.0,
                "comunidade": 1,
            },
            {
                "posicao": 3,
                "participante_id": "P003",
                "pagerank": 0.25,
                "grau": 1,
                "forca": 0.5,
                "betweenness": 0.0,
                "comunidade": 2,
            },
        ]
    ).to_csv(tabelas / "ranking_centralidade.csv", index=False)

    pd.DataFrame(
        [
            {
                "vertices": 3,
                "arestas": 2,
                "densidade": 0.6666666667,
                "componentes": 1,
                "maior_componente": 3,
                "grau_medio": 1.3333333333,
                "forca_media": 0.8,
                "peso_medio_arestas": 0.6,
                "agrupamento_medio": 0.0,
                "distancia_media_maior_componente": 0.4,
                "diametro_maior_componente": 0.5,
            }
        ]
    ).to_csv(tabelas / "resumo_grafo_final.csv", index=False)

    pd.DataFrame(
        [
            {
                "participantes_validos": 3,
                "jogos_validos": 2,
                "palpites_validos": 6,
                "resultados_validos": 2,
                "media_palpites_por_participante": 2.0,
                "mediana_palpites_por_participante": 2.0,
                "min_palpites_por_participante": 2,
                "max_palpites_por_participante": 2,
                "participantes_com_1_palpite": 0,
                "participantes_com_ate_4_palpites": 3,
            }
        ]
    ).to_csv(tabelas / "resumo_dados_artigo.csv", index=False)

    pd.DataFrame(
        [{"tamanho_componente": 3, "quantidade_componentes": 1, "quantidade_vertices_total": 3}]
    ).to_csv(tabelas / "resumo_componentes_artigo.csv", index=False)

    pd.DataFrame(
        [
            {
                "comunidade": 1,
                "tamanho": 2,
                "arestas_internas": 1,
                "densidade_interna": 1.0,
                "forca_media": 0.95,
                "pagerank_total": 0.75,
                "participante_mais_central": "P001",
                "pagerank_participante_mais_central": 0.4,
            },
            {
                "comunidade": 2,
                "tamanho": 1,
                "arestas_internas": 0,
                "densidade_interna": 0.0,
                "forca_media": 0.5,
                "pagerank_total": 0.25,
                "participante_mais_central": "P003",
                "pagerank_participante_mais_central": 0.25,
            },
        ]
    ).to_csv(tabelas / "resumo_comunidades_maior_componente.csv", index=False)

    pd.DataFrame(
        [[1.0, 0.7, 0.1], [0.7, 1.0, 0.5], [0.1, 0.5, 1.0]],
        index=["P001", "P002", "P003"],
        columns=["P001", "P002", "P003"],
    ).to_csv(tabelas / "matriz_similaridade_final.csv")

    for figura in [
        "analise_sensibilidade_limiar.png",
        "distribuicao_palpites_participante.png",
        "distribuicao_tamanho_componentes_melhorado.png",
        "tamanho_comunidades_maior_componente.png",
        "ranking_centralidade_pagerank_melhorado.png",
        "heatmap_similaridade_ordenado.png",
    ]:
        (figuras / figura).write_bytes(b"png-ficticio")


def test_reconstruir_grafo_final_a_partir_de_tabelas_minimas(tmp_path):
    _criar_saida_minima(tmp_path)

    G = reconstruir_grafo_final(str(tmp_path), theta=0.2)

    assert set(G.nodes) == {"P001", "P002", "P003"}
    assert G.number_of_edges() == 2
    assert G["P001"]["P002"]["weight"] == 0.7
    assert G["P001"]["P002"]["distance"] == pytest.approx(0.3)
    assert G.nodes["P001"]["posicao_ranking"] == 1
    assert G.nodes["P001"]["participante_id"] == "P001"


def test_exportar_grafo_graphml_cria_arquivo_nao_vazio(tmp_path):
    _criar_saida_minima(tmp_path)
    G = reconstruir_grafo_final(str(tmp_path), theta=0.2)
    caminho = tmp_path / "grafos" / "grafo_final.graphml"

    exportar_grafo_graphml(G, str(caminho))

    assert caminho.exists()
    assert caminho.stat().st_size > 0


def test_exportar_grafo_json_cria_metadata_nodes_edges(tmp_path):
    _criar_saida_minima(tmp_path)
    G = reconstruir_grafo_final(str(tmp_path), theta=0.2)
    caminho = tmp_path / "grafos" / "grafo_final.json"

    dados = exportar_grafo_json(G, str(caminho))

    assert caminho.exists()
    assert set(dados) == {"metadata", "nodes", "edges"}
    assert dados["metadata"]["vertices"] == 3
    assert dados["metadata"]["arestas"] == 2
    assert len(dados["nodes"]) == 3
    assert len(dados["edges"]) == 2


def test_gerar_relatorio_final_md_e_json(tmp_path):
    _criar_saida_minima(tmp_path)

    relatorio = gerar_relatorio_final(str(tmp_path), theta=0.2)

    assert (tmp_path / "resumo" / "relatorio_final.md").exists()
    assert (tmp_path / "resumo" / "relatorio_final.json").exists()
    assert relatorio["theta_experimental"] == 0.2
    assert "metricas_globais" in relatorio


def test_validacao_final_detecta_arquivos_essenciais(tmp_path):
    _criar_saida_minima(tmp_path)
    G = reconstruir_grafo_final(str(tmp_path), theta=0.2)
    exportar_grafo_graphml(G, str(tmp_path / "grafos" / "grafo_final.graphml"))
    exportar_grafo_json(G, str(tmp_path / "grafos" / "grafo_final.json"))
    gerar_relatorio_final(str(tmp_path), theta=0.2)

    resultado = gerar_validacao_final(str(tmp_path))

    assert resultado["ok"] is True
    assert (tmp_path / "resumo" / "validacao_final.txt").exists()
    assert (tmp_path / "resumo" / "validacao_final.json").exists()


def test_validacao_final_detecta_colunas_sensiveis(tmp_path):
    _criar_saida_minima(tmp_path)
    G = reconstruir_grafo_final(str(tmp_path), theta=0.2)
    exportar_grafo_graphml(G, str(tmp_path / "grafos" / "grafo_final.graphml"))
    exportar_grafo_json(G, str(tmp_path / "grafos" / "grafo_final.json"))
    gerar_relatorio_final(str(tmp_path), theta=0.2)
    pd.DataFrame([{"email": "pessoa@example.com", "participante_id": "P001"}]).to_csv(
        tmp_path / "tabelas" / "tabela_sensivel.csv",
        index=False,
    )

    resultado = gerar_validacao_final(str(tmp_path))

    assert resultado["ok"] is False
    assert any(check["nome"] == "dados_sensiveis" and not check["ok"] for check in resultado["checks"])


def test_grafo_json_contem_numero_esperado_de_nos_e_arestas(tmp_path):
    _criar_saida_minima(tmp_path)
    G = reconstruir_grafo_final(str(tmp_path), theta=0.2)
    caminho = tmp_path / "grafos" / "grafo_final.json"
    exportar_grafo_json(G, str(caminho))

    dados = json.loads(caminho.read_text(encoding="utf-8"))

    assert dados["metadata"]["vertices"] == 3
    assert dados["metadata"]["arestas"] == 2
