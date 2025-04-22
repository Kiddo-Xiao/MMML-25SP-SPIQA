#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch converter: strip LaTeX constructs from all **.txt** files in a folder tree
and write plain‑text versions to a destination folder, mirroring the directory
structure.

Adapted from https://github.com/art-r/tex-to-txt  (MIT License)
"""

import argparse
import re
from pathlib import Path
from typing import List


# ---------- 单文件处理核心 ---------- #
def read_file(path: Path) -> List[str]:
    """Read a LaTeX‑formatted text file, skipping comments and \\begin{...} blocks."""
    content, special = [], False
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.lstrip().startswith("%"):
                continue
            if special:
                if line.lstrip().startswith("\\end{"):
                    special = False
                continue
            if line.lstrip().startswith("\\begin{"):
                special = True
                continue
            content.append(line.rstrip("\n"))
    return content


def load_expressions(expr_path: str, add: bool) -> List[str]:
    defaults = (Path(__file__).with_name("defaults.txt")
                .read_text(encoding="utf-8").splitlines()
                if Path(__file__).with_name("defaults.txt").exists() else [])
    if expr_path == "default":
        return defaults
    custom = Path(expr_path).expanduser().read_text(encoding="utf-8").splitlines()
    return defaults + custom if add else custom + ["}"]


def strip(lines: List[str], exprs: List[str]) -> List[str]:
    rx = [
        r"\\footcite\[.*?]{.*?}", r"\\footnotetext{.*?}", r"\\footnote{.*?}",
        r"\\cite{.*?}", r"\\cite\[.*?]{.*?}", r"\\input{.*?}",
        r"\\ref{.*?}", r"\\label{.*?}",
    ]
    for r in rx:
        lines = [re.sub(r, "", ln) for ln in lines]
    for ex in exprs:
        lines = [ln.replace(ex, "") for ln in lines]
    return lines


def convert_one(src: Path, dst: Path, exprs: List[str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cleaned = strip(read_file(src), exprs)
    dst.write_text("\n".join(cleaned), encoding="utf-8")


# ---------- 批量 ---------- #
def batch(src_dir: Path, dst_dir: Path, exprs: List[str]) -> None:
    for src in src_dir.rglob("*.txt"):                       # ← 这里只看 .txt
        rel = src.relative_to(src_dir)
        convert_one(src, dst_dir / rel, exprs)
        print(f"✓ {src} → {dst_dir / rel}")


def parse():
    p = argparse.ArgumentParser(description="Batch clean LaTeX‑formatted .txt files.")
    p.add_argument("src_dir", help="directory containing latex‑formatted .txt files")
    p.add_argument("dst_dir", help="destination directory for cleaned files")
    p.add_argument("-e", "--expressions", default="default",
                   help="custom expressions file or 'default'")
    p.add_argument("-a", "--additional", action="store_true",
                   help="append custom expressions to defaults")
    return p.parse_args()


def main():
    args = parse()
    src = Path(args.src_dir).expanduser().resolve()
    dst = Path(args.dst_dir).expanduser().resolve()
    if not src.is_dir():
        raise NotADirectoryError(src)
    exprs = load_expressions(args.expressions, args.additional)
    batch(src, dst, exprs)


if __name__ == "__main__":
    main()
