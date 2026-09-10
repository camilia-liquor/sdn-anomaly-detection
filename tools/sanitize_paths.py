"""路径匿名化工具：把产物与文档里的本机绝对路径替换为占位符。

为什么需要
----------
实验产物的 JSON / 报告 / 日志里会记录本机绝对路径（例如 `C:\\Users\\<用户名>\\Desktop\\...`）。
竞赛规则明确要求评审材料中不得出现学校与个人身份信息；公开仓库同理不应暴露本机用户名。
本脚本把所有绝对路径统一替换为占位符，保证产物可公开而**不改变任何实验数值**。

用法：
    python tools/sanitize_paths.py                 # 预览将被修改的文件（dry-run）
    python tools/sanitize_paths.py --apply         # 实际替换
    python tools/sanitize_paths.py --apply --dirs results artifacts docs competition
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 只处理文本类文件
EXTS = {".json", ".md", ".txt", ".csv", ".log", ".py", ".bat", ".cfg", ".ini", ".yaml", ".yml"}
SKIP_DIRS = {".git", ".venv", "_tmp", ".cache", "__pycache__", "data", "node_modules"}

# 需要替换的模式（顺序重要：先替换项目根，再兜住其它用户目录）
PATTERNS = [
    (re.compile(re.escape(str(ROOT)), re.IGNORECASE), "<PROJECT_ROOT>"),
    (re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\\\s\"']+", re.IGNORECASE), "<USER_HOME>"),
    (re.compile(r"[A-Za-z]:\\Users\\[^\\\s\"']+", re.IGNORECASE), "<USER_HOME>"),
    (re.compile(r"[A-Za-z]:/Users/[^/\s\"']+", re.IGNORECASE), "<USER_HOME>"),
]


def sanitize_text(text: str) -> tuple[str, int]:
    n = 0
    for pat, repl in PATTERNS:
        text, k = pat.subn(repl, text)
        n += k
    return text, n


def iter_files(dirs: list[Path]):
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in EXTS:
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            yield p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="路径匿名化（把本机绝对路径替换为占位符）")
    ap.add_argument("--apply", action="store_true", help="实际写回；缺省只预览")
    ap.add_argument("--dirs", nargs="*", default=["results", "artifacts", "docs", "competition", "logs"],
                    help="要处理的目录（相对项目根）")
    args = ap.parse_args(argv)

    dirs = [ROOT / d for d in args.dirs]
    hit_files, hit_total = [], 0
    for p in iter_files(dirs):
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new, n = sanitize_text(text)
        if n:
            hit_files.append((p, n))
            hit_total += n
            if args.apply:
                p.write_text(new, encoding="utf-8")

    print(f"{'已替换' if args.apply else '待替换（预览）'}：{len(hit_files)} 个文件，共 {hit_total} 处路径")
    for p, n in hit_files[:25]:
        print(f"  {p.relative_to(ROOT)}  ({n} 处)")
    if len(hit_files) > 25:
        print(f"  ... 其余 {len(hit_files) - 25} 个文件省略")
    if not args.apply and hit_files:
        print("\n确认无误后执行： python tools/sanitize_paths.py --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
