"""Gera resultados do dashboard demo para os modelos linear e suavizado."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experimento_cobertura import (  # noqa: E402
    consolidar_visualizacoes_modelo_existente,
    executar_experimento_cobertura,
)
from src.pipeline import executar_pipeline  # noqa: E402


def gerar_resultados_demo(
    data_dir: str = "dashboard/data/demo",
    output_base: str = "outputs/dashboard_demo",
    theta: float = 0.20,
    gamma_suavizado: float = 0.5,
) -> dict:
    """Executa o pipeline demo nos modelos linear e suavizado."""
    base = Path(output_base)
    linear_dir = base / "linear"
    suavizado_dir = base / "suavizado"

    executar_pipeline(data_dir=data_dir, output_dir=str(linear_dir), theta=theta)
    consolidar_visualizacoes_modelo_existente(
        output_dir=str(linear_dir),
        data_dir=data_dir,
        theta=theta,
        gamma=1.0,
        modelo="cobertura_linear",
    )

    executar_experimento_cobertura(
        data_dir=data_dir,
        output_dir=str(suavizado_dir),
        theta=theta,
        gamma=gamma_suavizado,
        modelo_oficial_dir=str(linear_dir),
    )

    return {
        "data_dir": data_dir,
        "output_base": str(base),
        "linear_dir": str(linear_dir),
        "suavizado_dir": str(suavizado_dir),
        "theta": theta,
        "gamma_linear": 1.0,
        "gamma_suavizado": gamma_suavizado,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera resultados para o dashboard demo.")
    parser.add_argument("--data-dir", default="dashboard/data/demo")
    parser.add_argument("--output-base", default="outputs/dashboard_demo")
    parser.add_argument("--theta", type=float, default=0.20)
    parser.add_argument("--gamma-suavizado", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        resumo = gerar_resultados_demo(
            data_dir=args.data_dir,
            output_base=args.output_base,
            theta=args.theta,
            gamma_suavizado=args.gamma_suavizado,
        )
    except Exception as erro:
        print(f"Erro ao gerar resultados demo: {erro}", file=sys.stderr)
        return 1

    print("Resultados demo gerados com sucesso.")
    print(f"Modelo linear: {resumo['linear_dir']}")
    print(f"Modelo suavizado: {resumo['suavizado_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
