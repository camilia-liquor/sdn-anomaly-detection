"""真实数据集可复现下载器（多源回退 + 断点续传 + 校验清单）。

用法：
    python tools/fetch_datasets.py --list
    python tools/fetch_datasets.py --only cic_ids2017 ton_iot
    python tools/fetch_datasets.py --proxy http://127.0.0.1:7897
    python tools/fetch_datasets.py --verify          # 只校验已下载文件

说明：
* 数据集**不入版本库**（`data/real/` 在 .gitignore 中），本脚本是它们的可复现来源；
* 全部为公开学术数据集，仅用于研究评测，不再分发；引用信息写在 `data/real/MANIFEST.json`
  与 `docs/` 的清单文档里；
* 下载走显式代理（本机 Clash 127.0.0.1:7897；Python 不会自动读 Windows 系统代理），
  可用 `--proxy ""` 关闭。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "data" / "real"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

# ---------------------------------------------------------------- 数据集清单
# files 里的路径 = HF 仓库内相对路径；下载后用同样的相对结构落盘。
# expect_mb 仅用于在清单里标注预期体积，不参与校验（真实体积以实测为准）。
DATASETS: list[dict] = [
    {
        "name": "cic_ids2017",
        "title": "CIC-IDS2017（原始 MachineLearningCSV，逐日 CICFlowMeter 结果）",
        "repo": "bencorn/CICIDS2017",
        "files": ["csvs/MachineLearningCSV.zip", "csvs/GeneratedLabelledFlows.zip"],
        "extract": True,
        "license": "CIC / University of New Brunswick，学术研究用途；引用 Sharafaldin et al., 2018",
        "why": "逐日抓包（周一~周五）、BENIGN + 7 类攻击；列名与本工程 71 维特征族一致，"
               "是唯一同时支持「多类」与「按时序/在线」评测的公开集。"
               "MachineLearningCSV 是 78 特征压缩版（无时间列），GeneratedLabelledFlows 带 Timestamp，"
               "在线告警口径（检测时延/误报每小时）必须用后者",
    },
    {
        "name": "cic_ids2018",
        "title": "CSE-CIC-IDS2018（逐日 CICFlowMeter CSV，先取 3 天）",
        "repo": "c01dsnap/CIC-IDS2018",
        "files": [
            "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
            "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
            "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
        ],
        "license": "CIC / UNB + AWS，学术研究用途；引用 Sharafaldin et al., 2019",
        "why": "与 2017 同 schema 但实验床、攻击工具、年份都不同 → 检验「同族特征跨实验床」的迁移性",
    },
    {
        "name": "unsw_nb15",
        "title": "UNSW-NB15（官方训练/测试集 + 特征说明）",
        "repo": "Mireu-Lab/UNSW-NB15",
        "files": ["train.csv", "test.csv"],
        "license": "UNSW Canberra，学术研究用途；引用 Moustafa & Slay, 2015",
        "why": "特征体系完全不同（Zeek/Argus 系，49 维）→ 检验「换一套特征族还能不能用」，"
               "且官方自带 train/test 划分，避免自己切分带来的口径争议",
    },
    {
        "name": "ton_iot",
        "title": "TON_IoT（网络流量合并表）",
        "repo": "codymlewis/TON_IoT_network",
        "files": ["train_test_network.csv"],
        "license": "UNSW Canberra，学术研究用途；引用 Moustafa et al., 2020",
        "why": "IoT/IIoT 场景、含 label 与 type 两级标签、有时间列；与本工程的「校园网/企业网」场景形成对照",
    },
    {
        "name": "cic_iot2023",
        "title": "CIC-IoT2023（CICFlowMeter 逐攻击族 CSV，取 Benign + 7 族 DDoS）",
        "repo": "baalajimaestro/DDoS-CICIoT2023",
        "files": [
            "BenignTraffic.pcap.csv",
            "BenignTraffic1.pcap.csv",
            "DDoS-SYN_Flood.pcap.csv",
            "DDoS-UDP_Flood.pcap.csv",
            "DDoS-TCP_Flood.pcap.csv",
            "DDoS-ICMP_Flood.pcap.csv",
            "DDoS-PSHACK_Flood.pcap.csv",
            "DDoS-ICMP_Fragmentation.pcap.csv",
            "DDoS-UDP_Fragmentation.pcap.csv",
            "DDoS-SlowLoris.pcap.csv",
        ],
        "license": "CIC / UNB（CIC-IoT2023），学术研究用途；引用 Neto et al., 2023",
        "why": "同为 CICFlowMeter 特征族（能被本工程 71 维模型直接消费），但攻击族与 2017 完全不同，"
               "且含 SlowLoris / 分片洪泛这类低速隐蔽攻击 → 可在真实数据上做「训练未见过的攻击族」检验",
    },
]


# 镜像质量备注：核对出来的问题必须写进报告，不能因为「能读进来」就当成没事。
DATASET_NOTES: dict[str, list[str]] = {
    "cic_ids2017": [
        "MachineLearningCSV（78 特征版）与 GeneratedLabelledFlows（含 Timestamp 与流标识版）是"
        "**同一批流量的两种格式**，上表行数是两者之和，真实规模约为其一半；使用时只应二选一，不可合并计数。",
        "周四下午（Infiltration）当日存在缺失标签（`nan`），归类前必须剔除或单独处理，"
        "否则「19 类」里会多出一个假类别。",
        "标签字符串存在编码变体（如 Web Attack 的连字符在不同文件里不同），"
        "聚合成 8 大类前必须先做标签归一化。",
    ],
    "cic_ids2018": [
        "镜像里三个文件的样本数完全相同（都是 1,048,575 = 2^20−1，即 Excel 行数上限），"
        "疑似经表格工具导出时被截断；使用前应与原始发布方的行数核对，并按截断风险下调结论强度。",
    ],
    "unsw_nb15": [
        "该镜像的 train.csv = 82,332 行、test.csv = 175,341 行，与官方发布（训练 175,341 / 测试 82,332）相反，"
        "疑为文件名对调；引用具体数字前必须核对官方口径。",
        "特征为 Zeek/Argus 短名体系（45 列），与本工程 CICFlowMeter 71 维不通用（映射 0/71）。",
    ],
    "ton_iot": [
        "特征为 Argus/Zeek 体系（44 列），无时间列；与本工程 71 维不通用（映射 0/71），只能用于独立建模或标签级对照。",
    ],
    "cic_iot2023": [
        "数据里没有标签列，攻击族由文件名给出（工程按文件名推断标签并明确标注来源）。",
        "特征为 CICIoT2023 的 39 维（Header_Length / Tot sum / flag 计数 …），与本工程 71 维不通用（映射 0/71）。",
    ],
}


def _opener(proxy: str | None, insecure: bool):
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    if insecure:
        ctx = ssl._create_unverified_context()
    handlers = [urllib.request.HTTPSHandler(context=ctx)]
    if proxy:
        handlers.insert(0, urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    else:
        handlers.insert(0, urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)


def _url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def remote_size(opener, url: str, retries: int = 3) -> int | None:
    """探测远端文件总字节数（用 Range 请求读 Content-Range，避免拉全量）。"""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
            with opener.open(req, timeout=60) as r:
                cr = r.headers.get("Content-Range")
                if cr and "/" in cr:
                    return int(cr.rsplit("/", 1)[-1])
                cl = r.headers.get("Content-Length")
                return int(cl) if cl is not None else None
        except Exception:  # noqa: BLE001
            if attempt == retries:
                return None
            time.sleep(2 * attempt)
    return None


def download(opener, url: str, dest: Path, retries: int = 4, expected: int | None = None) -> dict:
    """断点续传下载；返回 {ok, bytes, attempts, error}。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if expected is None:
        expected = remote_size(opener, url)
    if dest.exists() and expected is not None and dest.stat().st_size >= expected:
        # 本地不短于远端：说明是上一次失败留下的可疑文件，重新下
        print(f"    本地 {dest.stat().st_size/1048576:.1f} MB ≥ 远端 {expected/1048576:.1f} MB，重新下载")
        dest.unlink()
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"User-Agent": UA}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with opener.open(req, timeout=90) as r:
                total = r.headers.get("Content-Length")
                total = int(total) + have if total else None
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                t0 = time.time()
                done = have
                with dest.open(mode) as f:
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        if total and done % (32 << 20) < (1 << 20):
                            pct = 100.0 * done / total
                            sp = (done - have) / max(time.time() - t0, 1e-6) / 1048576
                            print(f"    {dest.name}: {done/1048576:.1f}/{total/1048576:.1f} MB "
                                  f"({pct:.0f}%, {sp:.1f} MB/s)", flush=True)
            size = dest.stat().st_size
            if total and size < total:
                raise IOError(f"不完整：{size} < {total}")
            return {"ok": True, "bytes": size, "attempts": attempt}
        except Exception as e:  # noqa: BLE001
            if attempt == retries:
                return {"ok": False, "bytes": dest.stat().st_size if dest.exists() else 0,
                        "attempts": attempt, "error": f"{type(e).__name__}: {e}"}
            wait = 3 * attempt
            print(f"    第 {attempt} 次失败（{type(e).__name__}: {str(e)[:60]}），{wait}s 后重试…", flush=True)
            time.sleep(wait)
    return {"ok": False, "bytes": 0, "attempts": retries, "error": "unreachable"}


def _safe_member(name: str) -> str:
    """规范化 zip 成员名：Windows 会静默去掉路径分量结尾的空格/点，必须先剥掉，
    否则 CIC-IDS2017 的 `TrafficLabelling /` 目录建不出来（解压直接失败）。"""
    parts = [p.strip().rstrip(".") for p in name.replace("\\", "/").split("/")]
    return "/".join(p for p in parts if p not in ("", ".", ".."))


def extract_zip(zip_path: Path, dest_dir: Path) -> list[str]:
    names = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            rel = _safe_member(info.filename)
            if not rel:
                continue
            dest = dest_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, dest.open("wb") as out:
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    out.write(buf)
            names.append(rel)
    return names


# ---------------------------------------------------------------- 数据集体检
# 目的：对每份真实数据说清「能支撑哪种评测、不能支撑什么」，避免拿泄漏口径当成绩。
LABEL_HINTS = ("label", "target", "attack_cat", "attacktype", "class", "type")
LABEL_EXACT = {"label", "target", "attack_cat", "attack type", "class", "type",
               "category", "subcategory", "attacktype", "attack_type", "fine_grained_label"}
# 时间列必须**精确**匹配：子串匹配会命中 "Total Fwd Packets"（结尾 ts）、"spkts"、"Time_To_Live"，
# 从而凭空造出「这份数据能做在线评测」的假结论。
TIME_EXACT = ("timestamp", "time", "ts", "datetime", "date", "stime", "ltime",
              "flow start", "flow start time", "start time", "capture time")
TIME_HINTS = ("timestamp", "time", "stime", "ltime", "ts")
ID_HINTS = ("flow id", "src ip", "dst ip", "srcip", "dstip", "src_ip", "dst_ip",
            "sport", "dsport", "src port", "dst port", "src_port", "dst_port", "id")


def _read_csv_any(path: Path):
    import pandas as pd
    last = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc)
            df.columns = [str(c).strip() for c in df.columns]
            return df, enc
        except Exception as e:  # noqa: BLE001
            last = e
    raise last  # type: ignore[misc]


def _pick(cols: list[str], hints: tuple[str, ...]) -> str | None:
    """精确匹配（去空格、忽略大小写）；不做子串回退——子串匹配会误判时间列/标签列。"""
    low = {c.lower().strip(): c for c in cols}
    for h in hints:
        if h in low:
            return low[h]
    return None


def _load_build_mapping():
    """复用 tools/cross_dataset.py 里已验证的三级特征映射（保证与既有跨域口径一致）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_cd_mapping", ROOT / "tools" / "cross_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cd_mapping"] = mod
    spec.loader.exec_module(mod)
    return mod.build_mapping


def profile_all(out_root: Path) -> dict:
    import numpy as np
    import pandas as pd

    manifest_path = out_root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    meta_path = ROOT / "data" / "processed" / "meta.json"
    feature_names = json.loads(meta_path.read_text(encoding="utf-8"))["feature_names"] if meta_path.exists() else []
    build_mapping = _load_build_mapping() if feature_names else None

    report: dict = {"feature_reference": {"source": "data/processed/meta.json",
                                          "n_features": len(feature_names),
                                          "schema": "CICFlowMeter 全称命名"},
                    "datasets": {}}

    for name, rec in manifest.items():
        d = out_root / name
        csvs = sorted(p for p in d.rglob("*.csv"))
        if not csvs:
            continue
        print(f"\n=== 体检 {name}：{len(csvs)} 个 CSV")
        entry: dict = {"title": rec.get("title"), "repo": rec.get("repo"), "license": rec.get("license"),
                       "why": rec.get("why"), "files": []}
        all_cols: list[str] = []
        label_col = time_col = None
        label_counts: dict[str, int] = {}
        label_columns: dict[str, dict[str, int]] = {}   # 每个标签列各自的分布（二分类 + 攻击类型）
        total_rows = 0
        dup_rates = []
        time_ok = []
        numeric_cols_sample: list[str] = []

        for p in csvs:
            df, enc = _read_csv_any(p)
            total_rows += len(df)
            all_cols = list(df.columns) if len(df.columns) > len(all_cols) else all_cols
            lc = _pick(list(df.columns), LABEL_HINTS)
            tc = _pick(list(df.columns), TIME_EXACT)
            if lc and not label_col:
                label_col = lc
            if tc and not time_col:
                time_col = tc
            # 所有像标签的列都要统计（UNSW-NB15 的 label/attack_cat、TON_IoT 的 label/type）
            for c in df.columns:
                if c.lower().strip() in LABEL_EXACT:
                    dist = label_columns.setdefault(c, {})
                    for k, v in df[c].astype(str).value_counts().items():
                        dist[k] = dist.get(k, 0) + int(v)
            num = df.select_dtypes(include=[np.number])
            idlike = [c for c in num.columns if any(h in c.lower() for h in ID_HINTS)]
            feat = num.drop(columns=idlike, errors="ignore")
            if len(feat.columns) and len(df):
                dup_rates.append(float(feat.duplicated().mean()))
            if len(feat.columns) > len(numeric_cols_sample):
                numeric_cols_sample = list(feat.columns)
            # 没有标签列时，按文件名推断（CIC-IoT2023 的攻击族只在文件名里）
            inferred = None
            if lc is None:
                stem = p.name.split(".")[0]
                inferred = re.sub(r"\d+$", "", stem) or stem
                label_columns.setdefault("__from_filename__", {})
                label_columns["__from_filename__"][inferred] = \
                    label_columns["__from_filename__"].get(inferred, 0) + int(len(df))
            if tc:
                try:
                    ts = pd.to_datetime(df[tc], errors="coerce", format="mixed")
                    frac = float(ts.notna().mean())
                    mono = bool(ts.dropna().is_monotonic_increasing)
                    time_ok.append({"file": p.name, "parse_rate": round(frac, 4), "monotonic": mono})
                except Exception as e:  # noqa: BLE001
                    time_ok.append({"file": p.name, "error": str(e)[:60]})
            entry["files"].append({"name": p.name, "rows": int(len(df)), "cols": int(len(df.columns)),
                                   "encoding": enc, "label_col": lc, "time_col": tc})
            print(f'  {p.name[:52]:54} {len(df):>9,} 行 × {len(df.columns):>3} 列  标签={lc} 时间={tc}')

        entry["total_rows"] = total_rows
        entry["columns"] = all_cols
        # 主标签取「取值最多」的那一列（最细粒度）；二分类标签与细类标签都完整保留
        primary = max(label_columns, key=lambda c: len(label_columns[c])) if label_columns else label_col
        entry["label_col"] = primary
        entry["label_columns"] = {c: dict(sorted(d.items(), key=lambda kv: -kv[1]))
                                  for c, d in label_columns.items()}
        entry["label_distribution"] = entry["label_columns"].get(primary, {})
        entry["n_labels"] = len(entry["label_distribution"])
        entry["label_source"] = ("文件名推断（数据内无标签列）" if primary == "__from_filename__"
                                 else (f"数据列 `{primary}`" if primary else "无标签"))
        entry["notes"] = DATASET_NOTES.get(name, [])
        entry["time_col"] = time_col
        entry["time_detail"] = time_ok
        entry["exact_duplicate_rate"] = round(float(np.mean(dup_rates)), 4) if dup_rates else None

        if build_mapping and numeric_cols_sample:
            mapping, mp_info = build_mapping(feature_names, all_cols)
            entry["mapping"] = {
                "matched": len(mapping),
                "coverage": round(len(mapping) / max(len(feature_names), 1), 4),
                "method_counts": mp_info.get("method_counts") if isinstance(mp_info, dict) else None,
                "unmatched": [f for f in feature_names if f not in mapping][:15],
            }
            print(f'  特征映射：{len(mapping)}/{len(feature_names)}（{entry["mapping"]["coverage"]*100:.1f}%）')

        # 体检结论：能支撑哪些评测
        verdicts = []
        has_label = bool(entry["n_labels"])
        verdicts.append(("二分类（正常/攻击）", has_label, entry["label_source"] if has_label else "未识别到标签列"))
        multi = entry["n_labels"] > 2
        verdicts.append(("多分类", multi, f'标签取值 {entry["n_labels"]} 个'
                         + ("" if multi else "（只有 0/1，无法评多类）")))
        usable = bool(entry.get("mapping", {}).get("matched"))
        verdicts.append(("可直接用本工程 71 维模型", usable,
                         f'特征映射命中 {entry.get("mapping", {}).get("matched", 0)}/71'
                         + ("" if usable else " → 特征体系不同，需另建特征映射与模型，不能沿用现有权重")))
        temporal = bool(time_col) and any((t.get("parse_rate") or 0) > 0.99 for t in time_ok)
        ordered = bool(time_col) and any(t.get("monotonic") for t in time_ok)
        if not temporal:
            tnote = "无可用时间列 → 只能做静态窗口分类（不得报告检测时延/误报每小时）"
        elif ordered:
            tnote = "时间列可解析且（多数文件）行序即时间序 → 可直接做在线告警口径评测"
        else:
            tnote = "时间列可解析但**当前行序不是时间序** → 必须先按时间列排序再切分，否则时序评测无效"
        verdicts.append(("时序 / 在线评测（告警级）", temporal, tnote))
        dup = entry["exact_duplicate_rate"]
        leak = dup is None or dup > 0.2
        verdicts.append(("可做无泄漏切分", not leak,
                         f'精确重复率 {dup if dup is None else round(dup,3)}'
                         + ("（必须先按特征去重，否则同文件内切分必然泄漏）" if leak else "（重复率低）")))
        entry["verdicts"] = [{"aspect": a, "ok": bool(o), "note": n} for a, o, n in verdicts]
        report["datasets"][name] = entry

    return report


def write_profile(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "real_datasets.json"
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    L: list[str] = []
    ref = report["feature_reference"]
    L.append("# 真实数据集清单与体检（能支撑哪种评测）\n")
    L.append("> 由 `python tools/fetch_datasets.py --profile` 生成。数据集本身**不入版本库**"
             "（`data/real/` 已忽略），本文件是它们的来源、许可与「能/不能支撑什么评测」的记录。\n")
    L.append(f"特征基准：`{ref['source']}` 的 {ref['n_features']} 维（{ref['schema']}）。"
             "特征映射沿用 `tools/cross_dataset.py` 已验证的三级策略（显式映射表 → 同名 → 词元集合）。\n")
    L.append("\n## 1. 总览\n")
    L.append("| 数据集 | 行数 | 列数 | 标签数 | 时间列 | 精确重复率 | 特征映射覆盖 |")
    L.append("|---|---|---|---|---|---|---|")
    for name, e in report["datasets"].items():
        cov = e.get("mapping", {}).get("coverage")
        L.append(f"| `{name}` | {e['total_rows']:,} | {len(e['columns'])} | {e['n_labels']} | "
                 f"{e['time_col'] or '—'} | {e['exact_duplicate_rate']} | "
                 f"{(f'{cov*100:.1f}%' if cov is not None else '未评')} |")
    L.append("\n## 2. 逐数据集\n")
    for name, e in report["datasets"].items():
        L.append(f"### {name}　{e.get('title','')}\n")
        L.append(f"- 来源：Hugging Face `{e.get('repo')}`（镜像自原始发布方）")
        L.append(f"- 许可：{e.get('license')}")
        L.append(f"- 选它的理由：{e.get('why')}")
        L.append(f"- 规模：{e['total_rows']:,} 行 / {len(e['columns'])} 列，"
                 f"标签来源：{e.get('label_source')}（{e['n_labels']} 类），时间列 `{e['time_col'] or '无'}`")
        L.append(f"- 精确重复率（数值特征列）：**{e['exact_duplicate_rate']}**")
        m = e.get("mapping")
        if m:
            L.append(f"- 特征映射：**{m['matched']}/{ref['n_features']}（{m['coverage']*100:.1f}%）**"
                     + (f"，方法分布 {m['method_counts']}" if m.get("method_counts") else ""))
            if m.get("unmatched"):
                L.append(f"  - 未映射（前 15）：{'、'.join(m['unmatched'])}")
        L.append("- **能支撑的评测**：")
        for v in e["verdicts"]:
            L.append(f"  - {'✅' if v['ok'] else '❌'} {v['aspect']} —— {v['note']}")
        if e.get("notes"):
            L.append("- ⚠️ 镜像质量备注（用前必读）：")
            for n in e["notes"]:
                L.append(f"  - {n}")
        if e["label_columns"]:
            for c, dist in e["label_columns"].items():
                top = list(dist.items())[:12]
                L.append(f"- 标签 `{c}`（{len(dist)} 类）：" + "、".join(f"{k}={v:,}" for k, v in top)
                         + ("…" if len(dist) > 12 else ""))
        L.append("")
    L.append("\n## 3. 使用纪律\n")
    L.append("- 同文件内切分只有在**显式去重**后才可信；重复率高的数据集（见上表）必须先去重再切分；")
    L.append("- 没有单调时间列的数据集**不得**用于「在线告警/检测时延」口径的评测，只能做静态窗口分类；")
    L.append("- 跨数据集结论必须写明训练域与评估域，不能用同域数字冒充跨域能力；")
    L.append("- 数据集仅用于研究评测，不随仓库分发；引用信息见上表。")
    mp = out_dir / "real_datasets.md"
    mp.write_text("\n".join(L) + "\n", encoding="utf-8")
    return jp, mp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="真实数据集可复现下载器")
    ap.add_argument("--only", nargs="*", help="只处理指定数据集名")
    ap.add_argument("--out", default=str(OUT_ROOT), help="输出根目录（默认 data/real）")
    ap.add_argument("--proxy", default="http://127.0.0.1:7897",
                    help="显式代理；传空字符串关闭（Python 不读 Windows 系统代理）")
    ap.add_argument("--insecure", action="store_true", help="跳过 TLS 校验（默认开启校验）")
    ap.add_argument("--list", action="store_true", help="只列清单")
    ap.add_argument("--profile", action="store_true",
                    help="对已下载数据集做体检（标签/时间/重复率/特征映射覆盖），写 results/real_datasets.{json,md}")
    ap.add_argument("--verify", action="store_true", help="只校验已下载文件（重算 sha256）")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    ap.add_argument("--no-size-check", action="store_true",
                    help="不复核远端大小（默认会探测远端字节数，避免残缺文件被当成完整文件复用）")
    args = ap.parse_args(argv)

    if args.list:
        for d in DATASETS:
            print(f"{d['name']:14} {d['title']}")
            print(f"  来源   : Hugging Face {d['repo']}")
            print(f"  许可   : {d['license']}")
            print(f"  选它的理由: {d['why']}")
            print(f"  文件   : {', '.join(d['files'])}")
        return 0

    if args.profile:
        rep = profile_all(Path(args.out))
        jp, mp = write_profile(rep, ROOT / "results")
        print(f"\n体检报告已写出：\n  {jp}\n  {mp}")
        return 0

    proxy = args.proxy or None
    opener = _opener(proxy, args.insecure)
    out_root = Path(args.out)
    manifest_path = out_root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    todo = [d for d in DATASETS if not args.only or d["name"] in args.only]
    if not todo:
        print("没有匹配的数据集；用 --list 看清单")
        return 2

    print(f"代理：{proxy or '（直连）'}   输出：{out_root}")
    failures = []
    for d in todo:
        name = d["name"]
        print(f"\n=== {name}：{d['title']}")
        rec = manifest.setdefault(name, {"title": d["title"], "repo": d["repo"],
                                        "license": d["license"], "why": d["why"], "files": {}})
        rec.update({"title": d["title"], "repo": d["repo"], "license": d["license"], "why": d["why"]})
        for rel in d["files"]:
            dest = out_root / name / rel
            url = _url(d["repo"], rel)
            if args.verify:
                if not dest.exists():
                    print(f"  ✗ 缺失 {dest}")
                    failures.append(f"{name}/{rel}")
                    continue
                digest = _sha256(dest)
                rec["files"][rel] = {**rec["files"].get(rel, {}), "sha256": digest,
                                     "bytes": dest.stat().st_size, "url": url}
                print(f"  ✓ {rel}  {dest.stat().st_size/1048576:.1f} MB  sha256={digest[:16]}…")
                continue
            if dest.exists() and not args.force:
                local = dest.stat().st_size
                total = remote_size(opener, url) if not args.no_size_check else None
                if total is not None and local != total:
                    print(f"  ! 本地文件不完整（{local/1048576:.1f}/{total/1048576:.1f} MB），续传 {rel}")
                else:
                    digest = _sha256(dest)
                    rec["files"][rel] = {**rec["files"].get(rel, {}), "sha256": digest,
                                         "bytes": local, "url": url, "reused": True,
                                         "remote_bytes": total}
                    print(f"  = 已存在且完整，复用 {rel}（{local/1048576:.1f} MB）")
                    continue
            print(f"  ↓ {rel}")
            res = download(opener, url, dest)
            if res["ok"]:
                digest = _sha256(dest)
                rec["files"][rel] = {"sha256": digest, "bytes": res["bytes"], "url": url,
                                     "attempts": res["attempts"], "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                print(f"  ✓ 完成 {rel}  {res['bytes']/1048576:.1f} MB  sha256={digest[:16]}…")
            else:
                print(f"  ✗ 失败 {rel}  {res.get('error')}")
                failures.append(f"{name}/{rel}")
        if d.get("extract") and not args.verify:
            for rel in d["files"]:
                if not rel.lower().endswith(".zip"):
                    continue
                zp = out_root / name / rel
                if not zp.exists():
                    continue
                # 每个 zip 解到自己的子目录，否则第二个 zip 会被第一个的产物挡住而永不解开
                target = out_root / name / "extracted" / Path(rel).stem
                already = list(target.rglob("*.csv")) if target.exists() else []
                if already:
                    print(f"  = 已解压，跳过 {rel}（{len(already)} 个 CSV）")
                    rec.setdefault("extracted", {})[rel] = {"dir": str(target.relative_to(out_root)),
                                                            "files": [p.name for p in already]}
                    continue
                print(f"  ⇢ 解压 {rel} → {target}")
                names = extract_zip(zp, target)
                rec.setdefault("extracted", {})[rel] = {"dir": str(target.relative_to(out_root)),
                                                        "files": names}
                print(f"    解出 {len(names)} 个文件")
        manifest[name] = rec

    if not args.verify:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n清单已写入 {manifest_path}")

    if failures:
        print(f"\n有 {len(failures)} 个文件未成功：{', '.join(failures)}")
        return 1
    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
