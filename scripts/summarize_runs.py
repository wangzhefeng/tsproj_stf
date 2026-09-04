#!/usr/bin/env python3
"""汇总一组完整的多 seed 实验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsproj_stf.experiments.summary import summarize_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    summary = summarize_runs(args.root, args.prefix, seeds=tuple(args.seeds))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
