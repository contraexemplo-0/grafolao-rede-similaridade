"""Executa experimento complementar de cobertura suavizada."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experimento_cobertura import executar_experimento_cobertura  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa experimento de sensibilidade da penalizacao por cobertura."
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Diretorio com os CSVs canonicos anonimizados.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experimentos/cobertura_suavizada_gamma_0_5",
        help="Diretorio de saida exclusivo para o experimento complementar.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.20,
        help="Limiar de criacao de arestas.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Expoente aplicado a cobertura.",
    )
    parser.add_argument(
        "--modelo-oficial-dir",
        default="outputs/final_theta_0_20",
        help="Diretorio dos resultados oficiais usados na comparacao.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        resultado = executar_experimento_cobertura(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            theta=args.theta,
            gamma=args.gamma,
            modelo_oficial_dir=args.modelo_oficial_dir,
        )
    except Exception as erro:  # pragma: no cover - caminho de erro do CLI
        print(f"Erro no experimento de cobertura: {erro}", file=sys.stderr)
        return 1

    resumo = resultado["resumo"]
    print("Experimento de cobertura concluido.")
    print(f"Modelo: {resumo['modelo']}")
    print(f"Theta: {resumo['theta']}")
    print(f"Gamma: {resumo['gamma']}")
    print(f"Vertices: {resumo['vertices']}")
    print(f"Arestas: {resumo['arestas']}")
    print(f"Componentes: {resumo['componentes']}")
    print(f"Componentes unitarias: {resumo['componentes_unitarias']}")
    print(f"Maior componente: {resumo['maior_componente']}")
    print(f"Palpiteiro mais central: {resumo['palpiteiro_mais_central']}")
    print(f"Comparacao CSV: {resultado['caminhos']['comparacao_cobertura_csv']}")
    print(f"Comparacao PNG: {resultado['caminhos']['comparacao_cobertura_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
