"""Gera base demo do dashboard com o participante artificial P_TESTE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PARTICIPANTE_TESTE = "P_TESTE"


def _ler_csv(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo obrigatorio nao encontrado: {caminho}")
    return pd.read_csv(caminho, dtype={"participante_id": str, "jogo_id": str})


def _participante_base(palpites: pd.DataFrame, ranking_path: Path | None) -> str:
    if ranking_path is not None and ranking_path.exists():
        ranking = pd.read_csv(ranking_path, dtype={"participante_id": str})
        if "participante_id" in ranking.columns and not ranking.empty:
            for participante in ranking["participante_id"].tolist():
                if participante in set(palpites["participante_id"]):
                    return participante

    contagem = palpites.groupby("participante_id")["jogo_id"].nunique().sort_values(
        ascending=False
    )
    if contagem.empty:
        raise ValueError("Nao ha palpites disponiveis para gerar P_TESTE.")
    return str(contagem.index[0])


def _palpite_modal(palpites_jogo: pd.DataFrame) -> tuple[int, int]:
    pares = (
        palpites_jogo.groupby(["gols_a_palpite", "gols_b_palpite"])
        .size()
        .sort_values(ascending=False)
    )
    gols_a, gols_b = pares.index[0]
    return int(gols_a), int(gols_b)


def _ajustar_palpite(gols_a: int, gols_b: int, indice: int) -> tuple[int, int]:
    if indice % 7 in {0, 1, 2, 3}:
        return gols_a, gols_b
    if indice % 7 in {4, 5}:
        if gols_a >= gols_b:
            return max(0, gols_a + 1), gols_b
        return gols_a, max(0, gols_b + 1)

    if indice % 3 == 0:
        return 1, 1
    if indice % 3 == 1:
        return 2, 1
    return 1, 2


def gerar_dados_demo(
    input_dir: str = "data/processed",
    output_dir: str = "dashboard/data/demo",
    ranking_path: str | None = "outputs/final_theta_0_20/tabelas/ranking_centralidade.csv",
) -> dict:
    """Gera CSVs canonicos de demo sem alterar a base original."""
    entrada = Path(input_dir)
    saida = Path(output_dir)
    saida.mkdir(parents=True, exist_ok=True)

    participantes = _ler_csv(entrada / "participantes.csv")
    jogos = _ler_csv(entrada / "jogos.csv")
    palpites = _ler_csv(entrada / "palpites.csv")
    resultados = _ler_csv(entrada / "resultados.csv")

    if PARTICIPANTE_TESTE not in set(participantes["participante_id"]):
        participantes = pd.concat(
            [
                participantes,
                pd.DataFrame(
                    [
                        {
                            "participante_id": PARTICIPANTE_TESTE,
                            "rotulo": "Participante Teste",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    ranking = Path(ranking_path) if ranking_path else None
    participante_base = _participante_base(palpites, ranking)
    palpites_base = palpites[palpites["participante_id"] == participante_base]
    palpites_base_por_jogo = {
        linha.jogo_id: (int(linha.gols_a_palpite), int(linha.gols_b_palpite))
        for linha in palpites_base.itertuples(index=False)
    }

    jogos_validos = jogos["jogo_id"].tolist()
    if len(jogos_validos) > 10:
        jogos_p_teste = [jogo for indice, jogo in enumerate(jogos_validos) if indice != len(jogos_validos) - 1]
    else:
        jogos_p_teste = jogos_validos

    linhas_teste = []
    for indice, jogo_id in enumerate(jogos_p_teste):
        palpites_jogo = palpites[palpites["jogo_id"] == jogo_id]
        gols_a, gols_b = palpites_base_por_jogo.get(jogo_id, _palpite_modal(palpites_jogo))
        gols_a, gols_b = _ajustar_palpite(gols_a, gols_b, indice)
        registro = {
            "participante_id": PARTICIPANTE_TESTE,
            "jogo_id": jogo_id,
            "gols_a_palpite": int(gols_a),
            "gols_b_palpite": int(gols_b),
        }
        if "status_palpite" in palpites.columns:
            registro["status_palpite"] = "DEMO"
        if "pontos" in palpites.columns:
            registro["pontos"] = 0
        linhas_teste.append(registro)

    palpites_sem_teste = palpites[palpites["participante_id"] != PARTICIPANTE_TESTE].copy()
    palpites_demo = pd.concat([palpites_sem_teste, pd.DataFrame(linhas_teste)], ignore_index=True)
    palpites_demo = palpites_demo[palpites.columns]

    participantes.to_csv(saida / "participantes.csv", index=False)
    jogos.to_csv(saida / "jogos.csv", index=False)
    palpites_demo.to_csv(saida / "palpites.csv", index=False)
    resultados.to_csv(saida / "resultados.csv", index=False)

    return {
        "input_dir": str(entrada),
        "output_dir": str(saida),
        "participante_teste": PARTICIPANTE_TESTE,
        "participante_base": participante_base,
        "jogos_total": int(len(jogos_validos)),
        "palpites_p_teste": int(len(linhas_teste)),
        "participantes_total": int(len(participantes)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera dados demo anonimizados para o dashboard.")
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output-dir", default="dashboard/data/demo")
    parser.add_argument(
        "--ranking-path",
        default="outputs/final_theta_0_20/tabelas/ranking_centralidade.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        resumo = gerar_dados_demo(args.input_dir, args.output_dir, args.ranking_path)
    except Exception as erro:
        print(f"Erro ao gerar dados demo: {erro}", file=sys.stderr)
        return 1

    print("Dados demo gerados com sucesso.")
    print(f"Saida: {resumo['output_dir']}")
    print(f"Participante artificial: {resumo['participante_teste']}")
    print(f"Palpites de P_TESTE: {resumo['palpites_p_teste']} de {resumo['jogos_total']} jogos")
    print(f"Participante base usado: {resumo['participante_base']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
