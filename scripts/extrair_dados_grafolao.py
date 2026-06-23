"""CLI para extrair CSVs canonicos a partir do dump SQL do Grafolao."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extracao_sql import extrair_dados_grafolao


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrai e anonimiza CSVs canonicos do dump SQL do Grafolao.",
    )
    parser.add_argument("--input", required=True, help="Caminho do dump SQL de entrada.")
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Diretorio para os CSVs canonicos gerados.",
    )
    parser.add_argument(
        "--resumo-dir",
        default="outputs/resumo",
        help="Diretorio para resumo_extracao.json e resumo_extracao.txt.",
    )
    return parser


def main(argv=None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)

    try:
        resumo = extrair_dados_grafolao(
            sql_path=args.input,
            output_dir=args.output,
            resumo_dir=args.resumo_dir,
        )
    except Exception as erro:
        print(f"Erro ao extrair dados: {erro}", file=sys.stderr)
        return 1

    print("Extracao concluida com sucesso.")
    print(f"Jogos validos: {resumo['quantidade_jogos_validos']}")
    print(f"Palpites validos: {resumo['quantidade_palpites_validos']}")
    print(f"Participantes anonimizados: {resumo['quantidade_participantes_validos']}")
    print(f"CSVs gerados em: {args.output}")
    print(f"Resumo gerado em: {args.resumo_dir}")
    if resumo.get("avisos"):
        print(f"Avisos agregados: {len(resumo['avisos'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

