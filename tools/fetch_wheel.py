"""大 wheel 稳健下载器：支持断点续传 + 多镜像回退。

什么时候用得上
--------------
国内镜像（清华/阿里）在下载 200MB 以上的大 wheel（例如 ``tensorflow-intel``）时，
经常出现连接建立了但**一个字节都不写**、pip 一直卡住的情况。
本脚本改用"自己发 HTTP 请求 + 断点续传"的方式下载，慢但一定下得完：

    # 1) 先看要下哪个（pip 会打印真实 URL，或直接从 PyPI JSON API 取）
    python tools/fetch_wheel.py --resolve tensorflow-intel 2.15.1 --python 311 --plat win_amd64

    # 2) 下载到本地目录（中断后重跑会自动续传）
    python tools/fetch_wheel.py --out .cache/wheels tensorflow_intel-2.15.1-cp311-cp311-win_amd64.whl <url1> [url2 ...]

    # 3) 本地安装
    pip install --no-deps .cache/wheels/tensorflow_intel-2.15.1-cp311-cp311-win_amd64.whl

也可以用 ``--file`` 直接给文件名，或用 ``--resolve`` 一步查 URL 再下载。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_url(package: str, version: str, python_tag: str, plat: str) -> str | None:
    """从 PyPI JSON API 里挑出匹配的 wheel 下载地址。"""
    meta = json.loads(
        urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json", timeout=30).read()
    )
    for u in meta["urls"]:
        name = u["filename"]
        if name.endswith(".whl") and f"cp{python_tag}" in name and plat in name:
            return u["url"]
    for u in meta["urls"]:
        if u["filename"].endswith(".whl") and plat in u["filename"]:
            return u["url"]
    return None


def download(out_dir: str, name: str, urls: list[str], attempts: int = 4) -> int:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    part = path + ".part"
    for attempt, url in enumerate((urls * attempts)[: attempts * max(len(urls), 1)], 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"User-Agent": "pip/26", "Accept-Encoding": "identity"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length", 0)) + have
                mode = "ab" if have and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                log(f"尝试 {attempt}: 起始 {have / 1e6:.1f}MB  {url[:70]}…")
                last = time.time()
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if time.time() - last > 10:
                            log(f"  {have / 1e6:.1f} / {total / 1e6:.1f} MB  "
                                f"({have / max(time.time() - t0, 0.1) / 1e6:.2f} MB/s)")
                            last = time.time()
            if total and have >= total:
                os.replace(part, path)
                log(f"完成：{path}  {have / 1e6:.1f} MB")
                return 0
            log(f"  未下完（{have / 1e6:.1f}/{total / 1e6:.1f} MB），继续续传")
        except Exception as exc:  # noqa: BLE001
            log(f"  出错：{type(exc).__name__} {str(exc)[:100]}")
            time.sleep(3)
    log("多次尝试后仍未完成")
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="大 wheel 断点续传下载器")
    ap.add_argument("name", nargs="?", help="目标文件名（如 tensorflow_intel-2.15.1-cp311-cp311-win_amd64.whl）")
    ap.add_argument("urls", nargs="*", help="候选下载地址（可多个，按顺序回退）")
    ap.add_argument("--out", default=".cache/wheels", help="输出目录")
    ap.add_argument("--resolve", nargs=3, metavar=("PKG", "VER", "PYTAG"),
                    help="只查 URL 不下载，如 --resolve tensorflow-intel 2.15.1 311")
    ap.add_argument("--plat", default="win_amd64", help="平台标签，默认 win_amd64")
    args = ap.parse_args(argv)

    if args.resolve:
        pkg, ver, pytag = args.resolve
        url = resolve_url(pkg, ver, pytag, args.plat)
        if not url:
            log("没有找到匹配的 wheel")
            return 1
        print(url)
        return 0

    if not args.name or not args.urls:
        ap.print_help()
        return 1
    return download(args.out, args.name, list(args.urls))


if __name__ == "__main__":
    raise SystemExit(main())
