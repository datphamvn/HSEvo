"""Command-line interface for computing SWDI and CDI.

Examples::

    # From a JSON file containing a list of code strings
    python -m diversity --input snippets.json

    # From all Python files in a folder
    python -m diversity --folder ./problems/bpp_online --glob "*.py"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .core import DEFAULT_THRESHOLD, compute_diversity


def _load_from_json(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(
            f"{path} must contain a JSON array of code strings."
        )
    return data


def _load_from_folder(folder: Path, glob: str) -> List[str]:
    files = sorted(folder.rglob(glob))
    if not files:
        raise ValueError(f"No files matching '{glob}' found under {folder}.")
    return [f.read_text(encoding="utf-8") for f in files]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m diversity",
        description="Compute Shannon-Wiener (SWDI) and Cumulative (CDI) "
        "diversity indices for a set of code snippets.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        help="Path to a JSON file containing a list of code strings.",
    )
    source.add_argument(
        "--folder",
        type=Path,
        help="Path to a folder of source files to read as code snippets.",
    )
    parser.add_argument(
        "--glob",
        default="*.py",
        help="Glob pattern for --folder (default: '*.py').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Cosine-similarity threshold for SWDI (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional embedding model checkpoint override.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the embedding model (default: cpu).",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.input is not None:
            snippets = _load_from_json(args.input)
        else:
            snippets = _load_from_folder(args.folder, args.glob)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    embedder = None
    if args.checkpoint is not None:
        from .embeddings import CodeT5pEmbedder

        embedder = CodeT5pEmbedder(checkpoint=args.checkpoint, device=args.device)
    elif args.device != "cpu":
        from .embeddings import CodeT5pEmbedder

        embedder = CodeT5pEmbedder(device=args.device)

    result = compute_diversity(
        snippets, embedder=embedder, threshold=args.threshold
    )
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
