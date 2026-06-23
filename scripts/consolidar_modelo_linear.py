"""Consolida visualizacoes complementares do modelo linear oficial."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experimento_cobertura import consolidar_visualizacoes_modelo_existente  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera artefatos complementares para o modelo linear oficial."
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Diretorio com os CSVs canonicos anonimizados.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/final_theta_0_20",
        help="Diretorio com resultados oficiais ja calculados.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.20,
        help="Limiar experimental usado nos resultados oficiais.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Expoente de cobertura do modelo oficial.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        resultado = consolidar_visualizacoes_modelo_existente(
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            theta=args.theta,
            gamma=args.gamma,
            modelo="cobertura_linear",
        )
    except Exception as erro:  # pragma: no cover - caminho de erro do CLI
        print(f"Erro ao consolidar modelo linear: {erro}", file=sys.stderr)
        return 1

    print("Consolidacao do modelo linear concluida.")
    print(f"Output: {resultado['output_dir']}")
    print(f"Theta: {resultado['theta']}")
    print(f"Gamma: {resultado['gamma']}")
    print(f"Relatorio MD: {resultado['relatorio']['md']}")
    print(f"Relatorio JSON: {resultado['relatorio']['json']}")
    print(f"Metagrafo: {resultado['figuras']['metagrafo_comunidades']}")
    print(f"Vertices-ponte: {resultado['figuras']['vertices_ponte_intercomunidades']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
