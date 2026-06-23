"""CLI para analise de sensibilidade do limiar theta."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analise_limiar import THETAS_PADRAO, analisar_limiares


def _parse_thetas(valor: str | None) -> list[float] | None:
    if valor is None:
        return None

    partes = [parte.strip() for parte in valor.split(",")]
    if any(parte == "" for parte in partes):
        raise argparse.ArgumentTypeError("Informe thetas separados por virgula.")

    try:
        return [float(parte) for parte in partes]
    except ValueError as erro:
        raise argparse.ArgumentTypeError("Todos os thetas devem ser numericos.") from erro


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analisa a sensibilidade do grafo para diferentes limiares theta.",
    )
    parser.add_argument("--data-dir", required=True, help="Diretorio dos CSVs canonicos.")
    parser.add_argument("--output-dir", required=True, help="Diretorio de saida.")
    parser.add_argument(
        "--thetas",
        type=_parse_thetas,
        default=THETAS_PADRAO,
        help="Lista de limiares separados por virgula. Ex.: 0.2,0.3,0.4,0.5",
    )
    return parser


def main(argv=None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)

    try:
        resultado = analisar_limiares(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            thetas=args.thetas,
        )
    except Exception as erro:
        print(f"Erro ao analisar limiares: {erro}", file=sys.stderr)
        return 1

    tabela = resultado["tabela"]
    caminhos = resultado["caminhos"]

    print("Analise de limiar concluida com sucesso.")
    print(f"Thetas avaliados: {len(tabela)}")
    print(f"Vertices: {int(tabela['vertices'].iloc[0]) if not tabela.empty else 0}")
    print(f"Arestas por theta: {dict(zip(tabela['theta'], tabela['arestas']))}")
    print(f"Tabela: {caminhos['tabela']}")
    print(f"Resumo JSON: {caminhos['resumo_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

