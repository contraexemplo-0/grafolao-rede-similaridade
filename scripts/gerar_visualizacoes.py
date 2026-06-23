"""CLI para gerar visualizacoes em PNG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualizacao import gerar_visualizacoes


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera figuras PNG a partir das tabelas do pipeline.",
    )
    parser.add_argument("--output-dir", required=True, help="Diretorio de saida do pipeline.")
    parser.add_argument(
        "--theta",
        type=float,
        default=0.20,
        help="Limiar usado para reconstruir o grafo.",
    )
    return parser


def main(argv=None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)

    try:
        resultado = gerar_visualizacoes(output_dir=args.output_dir, theta=args.theta)
    except Exception as erro:
        print(f"Erro ao gerar visualizacoes: {erro}", file=sys.stderr)
        return 1

    print("Visualizacoes geradas com sucesso.")
    print(f"Vertices: {resultado['vertices']}")
    print(f"Arestas: {resultado['arestas']}")
    for nome, caminho in resultado["figuras"].items():
        print(f"{nome}: {caminho}")
    for aviso in resultado.get("avisos", []):
        print(f"Aviso: {aviso}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

