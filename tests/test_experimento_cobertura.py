from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from src.experimento_cobertura import (
    calcular_vertices_ponte_intercomunidades,
    consolidar_visualizacoes_modelo_existente,
    construir_metagrafo_comunidades,
    executar_experimento_cobertura,
    plotar_grafo_maior_componente_comunidades_organizado,
    plotar_grafo_pontes_intercomunidades,
    plotar_metagrafo_comunidades,
)
from src.pipeline import executar_pipeline
from src.similaridade import similaridade_participantes


def _criar_dados_canonicos(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"participante_id": "P001", "rotulo": "Participante 001"},
            {"participante_id": "P002", "rotulo": "Participante 002"},
            {"participante_id": "P003", "rotulo": "Participante 003"},
        ]
    ).to_csv(base / "participantes.csv", index=False)
    pd.DataFrame(
        [
            {"jogo_id": "J001", "rodada": 1},
            {"jogo_id": "J002", "rodada": 1},
        ]
    ).to_csv(base / "jogos.csv", index=False)
    pd.DataFrame(
        [
            {"participante_id": "P001", "jogo_id": "J001", "gols_a_palpite": 2, "gols_b_palpite": 1},
            {"participante_id": "P002", "jogo_id": "J001", "gols_a_palpite": 2, "gols_b_palpite": 1},
            {"participante_id": "P003", "jogo_id": "J001", "gols_a_palpite": 0, "gols_b_palpite": 1},
            {"participante_id": "P001", "jogo_id": "J002", "gols_a_palpite": 1, "gols_b_palpite": 0},
            {"participante_id": "P003", "jogo_id": "J002", "gols_a_palpite": 1, "gols_b_palpite": 0},
        ]
    ).to_csv(base / "palpites.csv", index=False)
    pd.DataFrame(
        [
            {"jogo_id": "J001", "gols_a_real": 2, "gols_b_real": 1, "status": "ENCERRADO"},
            {"jogo_id": "J002", "gols_a_real": 1, "gols_b_real": 0, "status": "ENCERRADO"},
        ]
    ).to_csv(base / "resultados.csv", index=False)


def _palpites_cobertura_parcial() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("P001", "J1", 2, 1),
            ("P002", "J1", 2, 1),
            ("P001", "J2", 2, 1),
            ("P002", "J2", 3, 1),
            ("P001", "J3", 0, 0),
            ("P002", "J3", 0, 1),
            ("P001", "J4", 1, 0),
        ],
        columns=["participante_id", "jogo_id", "gols_a_palpite", "gols_b_palpite"],
    )


def _grafo_com_duas_comunidades() -> nx.Graph:
    G = nx.Graph()
    atributos = {
        "P001": {"comunidade": 1, "pagerank": 0.3, "betweenness": 0.1},
        "P002": {"comunidade": 1, "pagerank": 0.2, "betweenness": 0.4},
        "P003": {"comunidade": 2, "pagerank": 0.25, "betweenness": 0.3},
        "P004": {"comunidade": 2, "pagerank": 0.15, "betweenness": 0.0},
    }
    for no, dados in atributos.items():
        G.add_node(no, **dados)
    G.add_edge("P001", "P002", weight=0.8, distance=0.2)
    G.add_edge("P002", "P003", weight=0.6, distance=0.4)
    G.add_edge("P003", "P004", weight=0.7, distance=0.3)
    G.add_edge("P001", "P003", weight=0.5, distance=0.5)
    return G


def test_gamma_um_reproduz_comportamento_anterior():
    palpites_df = _palpites_cobertura_parcial()
    jogos = ["J1", "J2", "J3", "J4"]

    padrao = similaridade_participantes("P001", "P002", palpites_df, jogos)
    gamma_um = similaridade_participantes(
        "P001",
        "P002",
        palpites_df,
        jogos,
        cobertura_gamma=1.0,
    )

    assert gamma_um["sim_final"] == pytest.approx(padrao["sim_final"])
    assert gamma_um["sim_final"] == pytest.approx(0.515625)


def test_gamma_meio_aumenta_similaridade_com_cobertura_parcial():
    palpites_df = _palpites_cobertura_parcial()
    jogos = ["J1", "J2", "J3", "J4"]

    linear = similaridade_participantes(
        "P001",
        "P002",
        palpites_df,
        jogos,
        cobertura_gamma=1.0,
    )
    suavizada = similaridade_participantes(
        "P001",
        "P002",
        palpites_df,
        jogos,
        cobertura_gamma=0.5,
    )

    assert suavizada["cobertura"] == pytest.approx(0.75)
    assert suavizada["sim_final"] >= linear["sim_final"]
    assert suavizada["sim_final"] == pytest.approx(0.6875 * (0.75**0.5))


def test_gamma_meio_nao_quebra_com_cobertura_zero():
    palpites_df = pd.DataFrame(
        [
            ("P001", "J1", 2, 1),
            ("P002", "J2", 2, 1),
        ],
        columns=["participante_id", "jogo_id", "gols_a_palpite", "gols_b_palpite"],
    )

    resultado = similaridade_participantes(
        "P001",
        "P002",
        palpites_df,
        ["J1", "J2"],
        cobertura_gamma=0.5,
    )

    assert resultado["cobertura"] == 0
    assert resultado["sim_final"] == 0


def test_experimento_gera_tabela_de_comparacao(tmp_path):
    data_dir = tmp_path / "data"
    oficial_dir = tmp_path / "oficial"
    experimento_dir = tmp_path / "experimentos" / "cobertura_suavizada_gamma_0_5"
    _criar_dados_canonicos(data_dir)
    executar_pipeline(str(data_dir), str(oficial_dir), theta=0.2)

    resultado = executar_experimento_cobertura(
        data_dir=str(data_dir),
        output_dir=str(experimento_dir),
        theta=0.2,
        gamma=0.5,
        modelo_oficial_dir=str(oficial_dir),
    )

    comparacao_path = tmp_path / "experimentos" / "comparacao_cobertura.csv"
    assert comparacao_path.exists()
    comparacao = pd.read_csv(comparacao_path)
    assert set(comparacao["modelo"]) == {"cobertura_linear", "cobertura_suavizada"}
    assert len(comparacao) == 2
    assert resultado["caminhos"]["comparacao_cobertura_csv"] == str(comparacao_path)

    arquivos_obrigatorios = [
        experimento_dir / "tabelas" / "vertices_ponte_intercomunidades.csv",
        experimento_dir / "tabelas" / "resumo_comunidades_apresentacao.csv",
        experimento_dir / "tabelas" / "resumo_dados_artigo.csv",
        experimento_dir / "figuras" / "grafo_maior_componente_comunidades_organizado.png",
        experimento_dir / "figuras" / "grafo_pontes_intercomunidades.png",
        experimento_dir / "figuras" / "metagrafo_comunidades.png",
        experimento_dir / "figuras" / "comparacao_cobertura.png",
        experimento_dir / "grafos" / "grafo_final.graphml",
        experimento_dir / "grafos" / "grafo_final.json",
        experimento_dir / "resumo" / "relatorio_experimento_cobertura.md",
        experimento_dir / "resumo" / "relatorio_experimento_cobertura.json",
    ]
    for arquivo in arquivos_obrigatorios:
        assert arquivo.exists()
        assert arquivo.stat().st_size > 0


def test_experimento_nao_sobrescreve_resultados_oficiais(tmp_path):
    data_dir = tmp_path / "data"
    oficial_dir = tmp_path / "oficial"
    experimento_dir = tmp_path / "experimentos" / "cobertura_suavizada_gamma_0_5"
    _criar_dados_canonicos(data_dir)
    executar_pipeline(str(data_dir), str(oficial_dir), theta=0.2)
    marcador = oficial_dir / "marcador.txt"
    marcador.write_text("resultado oficial preservado", encoding="utf-8")

    executar_experimento_cobertura(
        data_dir=str(data_dir),
        output_dir=str(experimento_dir),
        theta=0.2,
        gamma=0.5,
        modelo_oficial_dir=str(oficial_dir),
    )

    assert marcador.read_text(encoding="utf-8") == "resultado oficial preservado"
    assert not (oficial_dir / "comparacao_cobertura.csv").exists()


def test_calcula_grau_e_forca_intercomunidades():
    G = _grafo_com_duas_comunidades()

    pontes = calcular_vertices_ponte_intercomunidades(G)
    linha_p002 = pontes[pontes["participante_id"] == "P002"].iloc[0]
    linha_p001 = pontes[pontes["participante_id"] == "P001"].iloc[0]

    assert linha_p002["grau_intercomunidades"] == 1
    assert linha_p002["forca_intercomunidades"] == pytest.approx(0.6)
    assert linha_p001["grau_intercomunidades"] == 1
    assert linha_p001["forca_intercomunidades"] == pytest.approx(0.5)


def test_constroi_metagrafo_de_comunidades():
    G = _grafo_com_duas_comunidades()

    meta = construir_metagrafo_comunidades(G)

    assert set(meta.nodes) == {1, 2}
    assert meta.number_of_edges() == 1
    assert meta[1][2]["quantidade_arestas"] == 2
    assert meta[1][2]["peso_total"] == pytest.approx(1.1)
    assert meta[1][2]["peso_medio"] == pytest.approx(0.55)


def test_layout_organizado_por_comunidades_gera_png(tmp_path):
    G = _grafo_com_duas_comunidades()
    caminho = tmp_path / "grafo_organizado.png"

    plotar_grafo_maior_componente_comunidades_organizado(G, str(caminho))

    assert caminho.exists()
    assert caminho.stat().st_size > 0


def test_grafo_pontes_filtrado_gera_png(tmp_path):
    G = _grafo_com_duas_comunidades()
    pontes = calcular_vertices_ponte_intercomunidades(G)
    caminho = tmp_path / "grafo_pontes.png"

    plotar_grafo_pontes_intercomunidades(
        G,
        pontes,
        str(caminho),
        top_inter_edges=2,
        top_edge_labels=1,
    )

    assert caminho.exists()
    assert caminho.stat().st_size > 0


def test_metagrafo_gera_png(tmp_path):
    G = _grafo_com_duas_comunidades()
    meta = construir_metagrafo_comunidades(G)
    caminho = tmp_path / "metagrafo.png"

    plotar_metagrafo_comunidades(meta, str(caminho))

    assert caminho.exists()
    assert caminho.stat().st_size > 0


def test_consolida_artefatos_complementares_do_modelo_linear(tmp_path):
    data_dir = tmp_path / "data"
    oficial_dir = tmp_path / "oficial"
    _criar_dados_canonicos(data_dir)
    executar_pipeline(str(data_dir), str(oficial_dir), theta=0.2)
    resumo_antes = pd.read_csv(oficial_dir / "tabelas" / "resumo_grafo_final.csv")

    resultado = consolidar_visualizacoes_modelo_existente(
        output_dir=str(oficial_dir),
        data_dir=str(data_dir),
        theta=0.2,
        gamma=1.0,
        modelo="cobertura_linear",
    )

    arquivos = [
        oficial_dir / "tabelas" / "vertices_ponte_intercomunidades.csv",
        oficial_dir / "tabelas" / "resumo_comunidades_apresentacao.csv",
        oficial_dir / "figuras" / "grafo_maior_componente_comunidades_organizado.png",
        oficial_dir / "figuras" / "grafo_pontes_intercomunidades.png",
        oficial_dir / "figuras" / "metagrafo_comunidades.png",
        oficial_dir / "resumo" / "relatorio_experimento_modelo_linear.md",
        oficial_dir / "resumo" / "relatorio_experimento_modelo_linear.json",
    ]
    for arquivo in arquivos:
        assert arquivo.exists()
        assert arquivo.stat().st_size > 0

    resumo_depois = pd.read_csv(oficial_dir / "tabelas" / "resumo_grafo_final.csv")
    pd.testing.assert_frame_equal(resumo_antes, resumo_depois)
    assert resultado["caminhos"]["vertices_ponte_intercomunidades"] == str(arquivos[0])
