"""Extracao e anonimizacao do dump SQL real."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


TABELAS_RELEVANTES = {
    "public.usuarios",
    "public.times",
    "public.jogos",
    "public.palpites",
    "public.resultados",
}

COLUNAS_CSV = {
    "participantes": ["participante_id", "rotulo"],
    "jogos": [
        "jogo_id",
        "num",
        "rodada",
        "fase",
        "grupo",
        "time_a",
        "time_b",
        "data_hora",
        "status",
    ],
    "palpites": [
        "participante_id",
        "jogo_id",
        "gols_a_palpite",
        "gols_b_palpite",
        "status_palpite",
        "pontos",
    ],
    "resultados": ["jogo_id", "gols_a_real", "gols_b_real", "status"],
}

CONTAGENS_ESPERADAS_AUDITORIA = {
    "quantidade_jogos_validos": 33,
    "quantidade_palpites_validos": 1113,
    "quantidade_participantes_validos": 135,
}

COPY_RE = re.compile(
    r"^COPY\s+(public\.[A-Za-z_][\w]*)\s+\((.*?)\)\s+FROM\s+stdin;$"
)


def _limpar_coluna(coluna: str) -> str:
    return coluna.strip().strip('"')


def _converter_valor(valor: str) -> str | None:
    if valor == r"\N":
        return None
    return valor


def parse_copy_blocks(sql_path: str) -> dict:
    """
    Extrai blocos COPY de um dump SQL PostgreSQL.

    Args:
        sql_path (str): Caminho para o arquivo SQL.

    Returns:
        dict: Mapa por tabela contendo colunas e linhas como dicionarios.
    """
    caminho = Path(sql_path)
    if not caminho.exists():
        raise FileNotFoundError(f"Dump SQL nao encontrado: {sql_path}")

    blocos: dict[str, dict[str, Any]] = {}
    tabela_atual = None
    colunas_atuais: list[str] = []

    with caminho.open("r", encoding="utf-8", errors="replace") as arquivo:
        for linha_bruta in arquivo:
            linha = linha_bruta.rstrip("\n")

            if tabela_atual is None:
                match = COPY_RE.match(linha)
                if not match:
                    continue

                tabela_atual = match.group(1)
                colunas_atuais = [_limpar_coluna(c) for c in match.group(2).split(",")]
                blocos[tabela_atual] = {"columns": colunas_atuais, "rows": []}
                continue

            if linha == r"\.":
                tabela_atual = None
                colunas_atuais = []
                continue

            valores = linha.split("\t")
            registro = {
                coluna: _converter_valor(valores[indice]) if indice < len(valores) else None
                for indice, coluna in enumerate(colunas_atuais)
            }
            blocos[tabela_atual]["rows"].append(registro)

    return blocos


def _df_tabela(blocos: dict, tabela: str) -> pd.DataFrame:
    if tabela not in blocos:
        raise ValueError(f"Tabela obrigatoria ausente no dump: {tabela}")
    return pd.DataFrame(blocos[tabela]["rows"], columns=blocos[tabela]["columns"])


def _para_int_ou_none(valor):
    if pd.isna(valor):
        return None
    return int(valor)


def _ordenar_ids_numericos(df: pd.DataFrame) -> pd.DataFrame:
    ordenado = df.copy()
    ordenado["_num_sort"] = pd.to_numeric(ordenado["num"], errors="coerce")
    ordenado["_id_sort"] = ordenado["id"].astype(str)
    ordenado = ordenado.sort_values(
        by=["_num_sort", "_id_sort"],
        na_position="last",
        kind="mergesort",
    )
    return ordenado.drop(columns=["_num_sort", "_id_sort"])


def _resolver_time(row: pd.Series, time_map: dict[str, str], id_col: str, ref_col: str) -> str:
    time_id = row.get(id_col)
    referencia = row.get(ref_col)

    if pd.notna(time_id) and str(time_id) in time_map:
        return time_map[str(time_id)]
    if pd.notna(referencia) and str(referencia).strip():
        return str(referencia)
    return "INDEFINIDO"


def _usa_fallback_textual(
    row: pd.Series,
    time_map: dict[str, str],
    id_col: str,
    ref_col: str,
) -> bool:
    time_id = row.get(id_col)
    referencia = row.get(ref_col)
    id_ausente_ou_desconhecido = pd.isna(time_id) or str(time_id) not in time_map
    return bool(id_ausente_ou_desconhecido and pd.notna(referencia) and str(referencia).strip())


def _salvar_resumo(resumo: dict, resumo_dir: str) -> None:
    destino = Path(resumo_dir)
    destino.mkdir(parents=True, exist_ok=True)

    json_path = destino / "resumo_extracao.json"
    txt_path = destino / "resumo_extracao.txt"

    with json_path.open("w", encoding="utf-8") as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)

    linhas = ["Resumo da extracao do dump SQL", ""]
    for chave, valor in resumo.items():
        if chave == "avisos":
            continue
        linhas.append(f"{chave}: {valor}")

    linhas.append("")
    linhas.append("avisos:")
    for aviso in resumo.get("avisos", []):
        linhas.append(f"- {aviso}")

    txt_path.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def _adicionar_aviso_diferenca(resumo: dict, avisos: list[str]) -> None:
    for chave, esperado in CONTAGENS_ESPERADAS_AUDITORIA.items():
        obtido = resumo.get(chave)
        if obtido != esperado:
            avisos.append(f"{chave}: esperado na auditoria {esperado}, obtido {obtido}.")


def extrair_dados_grafolao(
    sql_path: str,
    output_dir: str = "data/processed",
    resumo_dir: str = "outputs/resumo",
) -> dict:
    """
    Extrai e anonimiza os CSVs canonicos a partir do dump SQL do Grafolao.

    Args:
        sql_path (str): Caminho para o dump SQL.
        output_dir (str): Diretorio dos CSVs canonicos de saida.
        resumo_dir (str): Diretorio dos resumos de extracao.

    Returns:
        dict: Resumo agregado da extracao, sem dados pessoais.
    """
    blocos = parse_copy_blocks(sql_path)
    tabelas_usadas = {tabela: blocos[tabela] for tabela in TABELAS_RELEVANTES if tabela in blocos}

    usuarios_df = _df_tabela(tabelas_usadas, "public.usuarios")
    times_df = _df_tabela(tabelas_usadas, "public.times")
    jogos_df = _df_tabela(tabelas_usadas, "public.jogos")
    palpites_df = _df_tabela(tabelas_usadas, "public.palpites")
    resultados_df = _df_tabela(tabelas_usadas, "public.resultados")

    jogos_encerrados = set(jogos_df.loc[jogos_df["status"] == "ENCERRADO", "id"])
    jogos_com_resultado = set(resultados_df["jogoId"])
    jogos_com_palpite = set(palpites_df["jogoId"])
    jogos_validos_ids = jogos_encerrados & jogos_com_resultado & jogos_com_palpite

    usuarios_existentes = set(usuarios_df["id"])
    palpites_validos_df = palpites_df[
        palpites_df["jogoId"].isin(jogos_validos_ids)
        & palpites_df["usuarioId"].isin(usuarios_existentes)
    ].copy()
    usuarios_validos_ids = sorted(str(uid) for uid in palpites_validos_df["usuarioId"].dropna().unique())

    participantes_map = {
        usuario_id: f"P{indice:03d}"
        for indice, usuario_id in enumerate(usuarios_validos_ids, start=1)
    }

    participantes_out = pd.DataFrame(
        [
            {
                "participante_id": participante_id,
                "rotulo": f"Participante {indice:03d}",
            }
            for indice, participante_id in enumerate(participantes_map.values(), start=1)
        ],
        columns=COLUNAS_CSV["participantes"],
    )

    jogos_validos_df = jogos_df[jogos_df["id"].isin(jogos_validos_ids)].copy()
    jogos_validos_df = _ordenar_ids_numericos(jogos_validos_df)
    jogos_map = {
        str(jogo_id): f"J{indice:03d}"
        for indice, jogo_id in enumerate(jogos_validos_df["id"], start=1)
    }

    time_map = {
        str(row["id"]): str(row["nome"])
        for _, row in times_df.iterrows()
        if pd.notna(row.get("id")) and pd.notna(row.get("nome"))
    }

    fallback_textual_mask = jogos_validos_df.apply(
        lambda row: _usa_fallback_textual(row, time_map, "timeCasaId", "timeCasaRef")
        or _usa_fallback_textual(row, time_map, "timeVisitanteId", "timeVisitanteRef"),
        axis=1,
    )

    jogos_out = pd.DataFrame(
        [
            {
                "jogo_id": jogos_map[str(row["id"])],
                "num": _para_int_ou_none(row.get("num")),
                "rodada": row.get("rodada"),
                "fase": row.get("fase"),
                "grupo": row.get("grupo"),
                "time_a": _resolver_time(row, time_map, "timeCasaId", "timeCasaRef"),
                "time_b": _resolver_time(row, time_map, "timeVisitanteId", "timeVisitanteRef"),
                "data_hora": row.get("dataHora"),
                "status": row.get("status"),
            }
            for _, row in jogos_validos_df.iterrows()
        ],
        columns=COLUNAS_CSV["jogos"],
    )

    palpites_validos_df["participante_id"] = palpites_validos_df["usuarioId"].map(participantes_map)
    palpites_validos_df["jogo_id"] = palpites_validos_df["jogoId"].map(jogos_map)
    palpites_out = pd.DataFrame(
        {
            "participante_id": palpites_validos_df["participante_id"],
            "jogo_id": palpites_validos_df["jogo_id"],
            "gols_a_palpite": pd.to_numeric(palpites_validos_df["golsCasa"], errors="raise").astype(int),
            "gols_b_palpite": pd.to_numeric(
                palpites_validos_df["golsVisitante"],
                errors="raise",
            ).astype(int),
            "status_palpite": palpites_validos_df["status"],
            "pontos": pd.to_numeric(palpites_validos_df["pontos"], errors="raise").astype(int),
        },
        columns=COLUNAS_CSV["palpites"],
    ).sort_values(["jogo_id", "participante_id"], kind="mergesort")

    resultados_validos_df = resultados_df[resultados_df["jogoId"].isin(jogos_validos_ids)].copy()
    resultados_validos_df["_jogo_id_canonico"] = resultados_validos_df["jogoId"].map(jogos_map)
    status_por_jogo = jogos_validos_df.set_index("id")["status"].to_dict()
    resultados_out = pd.DataFrame(
        {
            "jogo_id": resultados_validos_df["_jogo_id_canonico"],
            "gols_a_real": pd.to_numeric(resultados_validos_df["golsCasa"], errors="raise").astype(int),
            "gols_b_real": pd.to_numeric(
                resultados_validos_df["golsVisitante"],
                errors="raise",
            ).astype(int),
            "status": resultados_validos_df["jogoId"].map(status_por_jogo),
        },
        columns=COLUNAS_CSV["resultados"],
    ).sort_values(["jogo_id"], kind="mergesort")

    destino = Path(output_dir)
    destino.mkdir(parents=True, exist_ok=True)
    arquivos_gerados = {
        "participantes": str(destino / "participantes.csv"),
        "jogos": str(destino / "jogos.csv"),
        "palpites": str(destino / "palpites.csv"),
        "resultados": str(destino / "resultados.csv"),
    }

    participantes_out.to_csv(arquivos_gerados["participantes"], index=False)
    jogos_out.to_csv(arquivos_gerados["jogos"], index=False)
    palpites_out.to_csv(arquivos_gerados["palpites"], index=False)
    resultados_out.to_csv(arquivos_gerados["resultados"], index=False)

    jogos_encerrados_com_resultado = jogos_encerrados & jogos_com_resultado
    jogos_sem_palpite_excluidos = len(jogos_encerrados_com_resultado - jogos_com_palpite)
    palpites_usuario_inexistente = int(
        (
            palpites_df["jogoId"].isin(jogos_validos_ids)
            & ~palpites_df["usuarioId"].isin(usuarios_existentes)
        ).sum()
    )

    avisos: list[str] = []
    if jogos_sem_palpite_excluidos:
        avisos.append(
            f"{jogos_sem_palpite_excluidos} jogo(s) encerrado(s) com resultado e sem palpite foram excluidos."
        )
    quantidade_fallback = int(fallback_textual_mask.sum())
    if quantidade_fallback:
        avisos.append(f"{quantidade_fallback} jogo(s) usaram fallback textual de time.")
    if palpites_usuario_inexistente:
        avisos.append(
            f"{palpites_usuario_inexistente} palpite(s) de usuarios inexistentes foram excluidos."
        )

    resumo = {
        "arquivo_origem": str(sql_path),
        "formato_detectado": "COPY FROM stdin",
        "quantidade_usuarios_total": int(len(usuarios_df)),
        "quantidade_times_total": int(len(times_df)),
        "quantidade_jogos_total": int(len(jogos_df)),
        "quantidade_resultados_total": int(len(resultados_df)),
        "quantidade_palpites_total": int(len(palpites_df)),
        "quantidade_jogos_encerrados": int(len(jogos_encerrados)),
        "quantidade_jogos_com_resultado": int(len(jogos_com_resultado)),
        "quantidade_jogos_validos": int(len(jogos_validos_ids)),
        "quantidade_palpites_validos": int(len(palpites_out)),
        "quantidade_participantes_validos": int(len(participantes_out)),
        "quantidade_jogos_sem_time_id_com_fallback_textual": quantidade_fallback,
        "arquivos_gerados": arquivos_gerados,
        "avisos": avisos,
    }
    _adicionar_aviso_diferenca(resumo, avisos)
    _salvar_resumo(resumo, resumo_dir)

    return resumo

