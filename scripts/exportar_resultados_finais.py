"""Exporta artefatos finais do experimento principal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exportacao import (  # noqa: E402
    exportar_grafo_graphml,
    exportar_grafo_json,
    gerar_relatorio_final,
    reconstruir_grafo_final,
)
from src.validacao import gerar_validacao_final  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta GraphML, JSON, relatorio e validacao final do experimento."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/final_theta_0_20",
        help="Diretorio final contendo as tabelas e figuras do experimento.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.20,
        help="Limiar experimental usado para reconstruir o grafo final.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    grafos_dir = output_dir / "grafos"

    try:
        G = reconstruir_grafo_final(str(output_dir), theta=args.theta)
        graphml_path = grafos_dir / "grafo_final.graphml"
        json_path = grafos_dir / "grafo_final.json"

        exportar_grafo_graphml(G, str(graphml_path))
        exportar_grafo_json(G, str(json_path))
        gerar_relatorio_final(str(output_dir), theta=args.theta)
        validacao = gerar_validacao_final(str(output_dir))
    except Exception as erro:  # pragma: no cover - caminho de erro do CLI
        print(f"Erro na exportacao final: {erro}", file=sys.stderr)
        return 1

    print("Exportacao final concluida.")
    print(f"Vertices: {G.number_of_nodes()}")
    print(f"Arestas: {G.number_of_edges()}")
    print(f"GraphML: {graphml_path}")
    print(f"JSON do grafo: {json_path}")
    print(f"Relatorio Markdown: {output_dir / 'resumo' / 'relatorio_final.md'}")
    print(f"Relatorio JSON: {output_dir / 'resumo' / 'relatorio_final.json'}")
    print(f"Validacao final: {'OK' if validacao['ok'] else 'FALHA'}")
    print(f"Validacao TXT: {output_dir / 'resumo' / 'validacao_final.txt'}")
    print(f"Validacao JSON: {output_dir / 'resumo' / 'validacao_final.json'}")
    return 0 if validacao["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
