"""Command-line interface for SpecRule."""

import argparse
import json
from pathlib import Path

from .data import labeled_spectra, load_indian_pines
from .plotting import plot_best_rule
from .search import discover_class_rules


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = json.loads(args.config.read_text())
    cube, labels = load_indian_pines(config["data_path"], config["labels_path"])
    spectra, target = labeled_spectra(cube, labels)
    rules = discover_class_rules(spectra, target, max_pairs=args.max_pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rules.to_csv(args.output_dir / "discovered_rules.csv", index=False)
    plot_best_rule(spectra, target, rules, args.output_dir / "rule_diagnostic.png")
    print(rules.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
