from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.extracao_sql import extrair_dados_grafolao, parse_copy_blocks


def _dump_ficticio() -> str:
    return """COPY public.usuarios (id, nome, email, "avatarUrl", "googleId", role, ativo, "criadoEm", "atualizadoEm", "comunidadesCriadas") FROM stdin;
u2\tPessoa Dois\tdois@example.test\tavatar-2\tgoogle-2\tPARTICIPANTE\ttrue\t2026-01-01\t2026-01-01\t0
u1\tPessoa Um\tum@example.test\tavatar-1\tgoogle-1\tPARTICIPANTE\ttrue\t2026-01-01\t2026-01-01\t0
u3\tPessoa Tres\ttres@example.test\tavatar-3\tgoogle-3\tPARTICIPANTE\ttrue\t2026-01-01\t2026-01-01\t0
\\.
COPY public.times (id, nome, codigo, grupo) FROM stdin;
t1\tTime A\tTMA\tA
t2\tTime B\tTMB\tB
\\.
COPY public.jogos (id, num, fase, rodada, grupo, "dataHora", local, "timeCasaId", "timeVisitanteId", "timeCasaRef", "timeVisitanteRef", status) FROM stdin;
g1\t1\tGRUPOS\t1\tA\t2026-06-01 12:00:00\tLocal\tt1\tt2\t\\N\t\\N\tENCERRADO
g2\t2\tGRUPOS\t1\tA\t2026-06-02 12:00:00\tLocal\tt1\tt2\t\\N\t\\N\tENCERRADO
g3\t3\tGRUPOS\t2\tA\t2026-06-03 12:00:00\tLocal\tt1\tt2\t\\N\t\\N\tENCERRADO
g4\t4\tGRUPOS\t2\tA\t2026-06-04 12:00:00\tLocal\t\\N\t\\N\tTime Fallback A\tTime Fallback B\tENCERRADO
g5\t5\tGRUPOS\t3\tA\t2026-06-05 12:00:00\tLocal\tt1\tt2\t\\N\t\\N\tAGENDADO
\\.
COPY public.palpites (id, "usuarioId", "jogoId", "resultadoId", "golsCasa", "golsVisitante", pontos, status, "criadoEm", "atualizadoEm", "totalEdicoes") FROM stdin;
p1\tu2\tg1\tr1\t2\t1\t3\tACERTO_VENCEDOR\t2026-01-01\t2026-01-01\t0
p2\tu1\tg1\tr1\t1\t1\t0\tERRO\t2026-01-01\t2026-01-01\t0
p3\tu2\tg2\t\\N\t0\t0\t0\tPENDENTE\t2026-01-01\t2026-01-01\t0
p4\tu1\tg4\tr4\t3\t2\t5\tACERTO_PLACAR\t2026-01-01\t2026-01-01\t0
p5\tu1\tg5\tr5\t1\t0\t0\tPENDENTE\t2026-01-01\t2026-01-01\t0
\\.
COPY public.resultados (id, "jogoId", "golsCasa", "golsVisitante", "cartoesAmarelos", "cartoesVermelhosIndiretos", "cartoesVermelhosDiretos", "cartoesAmarelosMaisVermelho", "inseridoEm", "inseridoPor", "artilheirosCasa", "artilheirosVisitante") FROM stdin;
r1\tg1\t2\t0\t0\t0\t0\t0\t2026-01-01\tadmin\t{}\t{}
r3\tg3\t1\t1\t0\t0\t0\t0\t2026-01-01\tadmin\t{}\t{}
r4\tg4\t3\t2\t0\t0\t0\t0\t2026-01-01\tadmin\t{}\t{}
r5\tg5\t1\t0\t0\t0\t0\t0\t2026-01-01\tadmin\t{}\t{}
\\.
"""


def _gravar_dump(tmp_path: Path) -> Path:
    caminho = tmp_path / "dump_ficticio.sql"
    caminho.write_text(_dump_ficticio(), encoding="utf-8")
    return caminho


def test_parse_copy_blocks_detecta_tabela_colunas_linhas_e_null(tmp_path):
    dump = tmp_path / "dump.sql"
    dump.write_text(
        "COPY public.teste (id, nome, valor) FROM stdin;\n"
        "1\tTexto\t\\N\n"
        "\\.\n",
        encoding="utf-8",
    )

    blocos = parse_copy_blocks(str(dump))

    assert "public.teste" in blocos
    assert blocos["public.teste"]["columns"] == ["id", "nome", "valor"]
    assert blocos["public.teste"]["rows"] == [
        {"id": "1", "nome": "Texto", "valor": None}
    ]


def test_extrair_dados_grafolao_gera_csvs_canonicos_anonimizados(tmp_path):
    dump = _gravar_dump(tmp_path)
    output_dir = tmp_path / "processed"
    resumo_dir = tmp_path / "resumo"

    resumo = extrair_dados_grafolao(
        sql_path=str(dump),
        output_dir=str(output_dir),
        resumo_dir=str(resumo_dir),
    )

    participantes = pd.read_csv(output_dir / "participantes.csv")
    jogos = pd.read_csv(output_dir / "jogos.csv")
    palpites = pd.read_csv(output_dir / "palpites.csv")
    resultados = pd.read_csv(output_dir / "resultados.csv")

    assert set(participantes.columns) == {"participante_id", "rotulo"}
    assert participantes["participante_id"].tolist() == ["P001", "P002"]
    assert participantes["rotulo"].tolist() == ["Participante 001", "Participante 002"]
    assert not {"id", "nome", "email", "avatarUrl", "googleId"}.intersection(
        participantes.columns
    )

    assert jogos["jogo_id"].tolist() == ["J001", "J002"]
    assert set(palpites["jogo_id"]) == {"J001", "J002"}
    assert set(resultados["jogo_id"]) == {"J001", "J002"}
    assert set(resultados["status"]) == {"ENCERRADO"}

    assert "usuarioId" not in palpites.columns
    assert set(palpites["participante_id"]) == {"P001", "P002"}

    assert "J003" not in set(jogos["jogo_id"])
    assert resumo["quantidade_jogos_validos"] == 2
    assert resumo["quantidade_palpites_validos"] == 3
    assert resumo["quantidade_participantes_validos"] == 2


def test_extracao_exclui_jogos_sem_resultado_e_sem_palpite(tmp_path):
    dump = _gravar_dump(tmp_path)
    output_dir = tmp_path / "processed"
    resumo_dir = tmp_path / "resumo"

    extrair_dados_grafolao(str(dump), str(output_dir), str(resumo_dir))

    jogos = pd.read_csv(output_dir / "jogos.csv")
    palpites = pd.read_csv(output_dir / "palpites.csv")

    assert jogos["num"].tolist() == [1, 4]
    assert set(palpites["jogo_id"]) == {"J001", "J002"}


def test_extracao_usa_fallback_textual_para_times(tmp_path):
    dump = _gravar_dump(tmp_path)
    output_dir = tmp_path / "processed"
    resumo_dir = tmp_path / "resumo"

    resumo = extrair_dados_grafolao(str(dump), str(output_dir), str(resumo_dir))
    jogos = pd.read_csv(output_dir / "jogos.csv")
    jogo_fallback = jogos.loc[jogos["jogo_id"] == "J002"].iloc[0]

    assert jogo_fallback["time_a"] == "Time Fallback A"
    assert jogo_fallback["time_b"] == "Time Fallback B"
    assert resumo["quantidade_jogos_sem_time_id_com_fallback_textual"] == 1


def test_resumo_extracao_eh_gerado_sem_valores_pessoais(tmp_path):
    dump = _gravar_dump(tmp_path)
    output_dir = tmp_path / "processed"
    resumo_dir = tmp_path / "resumo"

    extrair_dados_grafolao(str(dump), str(output_dir), str(resumo_dir))

    resumo_json = resumo_dir / "resumo_extracao.json"
    resumo_txt = resumo_dir / "resumo_extracao.txt"
    assert resumo_json.exists()
    assert resumo_txt.exists()

    resumo = json.loads(resumo_json.read_text(encoding="utf-8"))
    assert "arquivos_gerados" in resumo

    conteudo_saida = "\n".join(
        caminho.read_text(encoding="utf-8")
        for caminho in [
            output_dir / "participantes.csv",
            output_dir / "jogos.csv",
            output_dir / "palpites.csv",
            output_dir / "resultados.csv",
            resumo_json,
            resumo_txt,
        ]
    )
    for valor_pessoal in [
        "Pessoa Um",
        "Pessoa Dois",
        "Pessoa Tres",
        "um@example.test",
        "dois@example.test",
        "tres@example.test",
        "avatar-1",
        "google-1",
        "u1",
        "u2",
        "u3",
    ]:
        assert valor_pessoal not in conteudo_saida

