"""Interface de linha de comando do pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import THETA_SIMILARIDADE
from src.pipeline import executar_pipeline


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa o pipeline da Rede de Similaridade de Palpites.",
    )
    parser.add_argument("--data-dir", required=True, help="Diretorio dos CSVs canonicos.")
    parser.add_argument("--output-dir", required=True, help="Diretorio de saida.")
    parser.add_argument(
        "--theta",
        type=float,
        default=THETA_SIMILARIDADE,
        help="Limiar de similaridade para criacao de arestas.",
    )
    return parser


def main(argv=None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)

    try:
        resultado = executar_pipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            theta=args.theta,
        )
    except Exception as erro:
        print(f"Erro ao executar pipeline: {erro}", file=sys.stderr)
        return 1

    resumo = resultado["resumo"]
    print("Pipeline executado com sucesso.")
    print(f"Participantes: {resumo['quantidade_participantes']}")
    print(f"Jogos: {resumo['quantidade_jogos']}")
    print(f"Arestas do grafo final: {resumo['arestas']}")
    print(f"Palpiteiro mais central: {resumo['palpiteiro_mais_central']}")
    print(f"Saidas em: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
