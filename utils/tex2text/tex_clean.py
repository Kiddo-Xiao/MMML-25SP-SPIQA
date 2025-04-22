#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量删除 LaTeX 图片与表格，并将结果保存到新文件夹
用法:
    python strip_fig_table.py /path/to/txts            # 输出到 /path/to/txts/cleaned
    python strip_fig_table.py /path/to/txts --out out  # 输出到 /path/to/txts/out
"""

import re
import argparse
import pathlib

# ---------- 核心过滤函数 ----------
def strip_latex_blocks(text: str) -> str:
    """移除 figure、table、tabular 环境及 \includegraphics 命令"""
    for env in ("figure", "table"):
        text = re.sub(
            rf"\\begin{{{env}}}.*?\\end{{{env}}}",
            "",
            text,
            flags=re.DOTALL,
        )
    text = re.sub(r"\\begin{tabular}.*?\\end{tabular}", "", text, flags=re.DOTALL)
    text = re.sub(
        r"\\includegraphics(?:\[[^\]]*])?{[^}]*}",
        "",
        text,
    )
    return text

# ---------- 主流程 ----------
def main(src_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)          # 若不存在则创建
    for src_file in src_dir.glob("*.txt"):
        cleaned_text = strip_latex_blocks(src_file.read_text(encoding="utf-8", errors="ignore"))
        dst_file = out_dir / src_file.name              # 保持原文件名
        dst_file.write_text(cleaned_text, encoding="utf-8")
        print(f"done {src_file.name}  ➜  {dst_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove LaTeX figures/tables from .txt files and save to a new folder."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="待处理目录（默认当前目录）",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="cleaned",
        help="输出文件夹名称或路径(默认 'cleaned'—)",
    )
    args = parser.parse_args()

    src_path = pathlib.Path(args.directory).expanduser().resolve()
    out_path = (src_path / args.out).resolve() if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out).resolve()

    main(src_path, out_path)
