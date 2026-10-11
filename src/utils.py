"""通用工具：项目路径、随机种子、日志、JSON/CSV 读写、matplotlib 中文字体。

所有模块都从这里取路径，保证「在任何机器上 clone 下来就能跑」——
原始 demo 里硬编码的 C:\\Users\\jiayu\\... 路径在这里被彻底消除。
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# --------------------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
RESULT_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = RESULT_DIR / "figures"
LOG_DIR = PROJECT_ROOT / "logs"

# --------------------------------------------------------------------------------------
# 把第三方库的缓存目录也收进项目内（这样"整份工程拷走就能跑"，不依赖用户主目录的写权限）
#   必须在 import matplotlib / tensorflow 之前设置好。
# --------------------------------------------------------------------------------------
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".cache" / "keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")       # 少刷 TF 的 INFO 日志
os.environ.setdefault("PYTHONIOENCODING", "utf-8")        # Windows 控制台中文
for _d in (Path(os.environ["MPLCONFIGDIR"]), Path(os.environ["KERAS_HOME"])):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

_LOGGER_READY: set[str] = set()


def ensure_dirs(*dirs: Path | str) -> None:
    """创建目录（幂等）。"""
    targets: Iterable[Path | str] = dirs or (
        RAW_DIR,
        PROCESSED_DIR,
        ARTIFACT_DIR,
        FIGURE_DIR,
        LOG_DIR,
    )
    for d in targets:
        Path(d).mkdir(parents=True, exist_ok=True)


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def resolve_dir(value: str | Path | None, default: Path) -> Path:
    """把配置里的目录字符串解析成绝对路径（空值则用默认目录）。

    用于让不同实验（正式实验 / 快跑自检）各自使用**独立的数据目录**，
    避免快跑把正式实验的 data/raw、data/processed 覆盖掉。
    """
    if value in (None, "", False):
        return Path(default)
    p = Path(str(value))
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# --------------------------------------------------------------------------------------
# 随机种子 / 复现性
# --------------------------------------------------------------------------------------
def silence_tf_logs() -> None:
    """压掉 TensorFlow / absl 的告警刷屏，让实验日志保持干净可读。"""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*deprecated.*")
    warnings.filterwarnings("ignore", message=".*tf\\.executing_eagerly.*")
    try:
        import absl.logging as _absl

        _absl.set_verbosity(_absl.ERROR)
        logging.getLogger("absl").setLevel(logging.ERROR)
    except Exception:
        pass


def set_seed(seed: int = 42) -> None:
    """固定所有随机源（Python / NumPy / TensorFlow）。

    TensorFlow 是可选依赖：没安装时只固定 Python 与 NumPy 的种子，
    保证数据生成与切分依然完全可复现。
    """
    silence_tf_logs()
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - 取决于是否装了 TF
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass
        # 让 TF 在 CPU 上尽量确定性（GPU 上会略降速，比赛跑分时可用 --no-deterministic 关掉）
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# 日志
# --------------------------------------------------------------------------------------
def get_logger(name: str = "cida", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in _LOGGER_READY:
        # 同名 logger 已经建过：如果这次带了新的日志文件，把文件 handler 补上，
        # 否则会出现"日志目录是空的"这种坑。
        if log_file is not None:
            _attach_file_handler(logger, log_file)
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    stream = sys.stdout
    # Windows 控制台默认 GBK，中文日志会炸；尽量切到 UTF-8
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    sh = logging.StreamHandler(stream)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        _attach_file_handler(logger, log_file, fmt)

    _LOGGER_READY.add(name)
    return logger


def _attach_file_handler(logger: logging.Logger, log_file: str | Path, fmt: logging.Formatter | None = None) -> None:
    path = Path(log_file).resolve()
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")).resolve() == path:
            return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt or logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)


def banner(logger: logging.Logger, text: str, char: str = "=") -> None:
    logger.info(char * 78)
    logger.info(text)
    logger.info(char * 78)


# --------------------------------------------------------------------------------------
# JSON / 文本读写
# --------------------------------------------------------------------------------------
class _NpEncoder(json.JSONEncoder):
    def default(self, o: Any):  # noqa: D401
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=_NpEncoder)
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
    return path


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


# --------------------------------------------------------------------------------------
# matplotlib 中文
# --------------------------------------------------------------------------------------
_CJK_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",     # 黑体
    "C:/Windows/Fonts/simsun.ttc",     # 宋体
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def setup_matplotlib(chinese: bool = True, logger: logging.Logger | None = None) -> bool:
    """配置 matplotlib：无界面后端 + 可用则启用中文字体。

    返回 True 表示中文可用（图表用中文标注），False 表示回退到英文标注。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["figure.facecolor"] = "white"

    if not chinese:
        return False

    from matplotlib import font_manager

    for cand in _CJK_CANDIDATES:
        if Path(cand).exists():
            try:
                font_manager.fontManager.addfont(cand)
                name = font_manager.FontProperties(fname=cand).get_name()
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                if logger:
                    logger.info("绘图字体已启用：%s", name)
                return True
            except Exception:
                continue
    if logger:
        logger.warning("未找到中文字体，图表将使用英文标注（可把字体文件放到系统字体目录后重跑）。")
    return False


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------
def timestamp_tag() -> str:
    import datetime as _dt

    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def fmt_float(x: float, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """把二维数据渲染成 markdown 表格（用于自动生成实验报告）。"""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out)


def _rotate_jsonl(p: Path, backups: int) -> None:
    """把 p 依 `<name>.1` / `<name>.2` … 滚动，删掉超出 backups 的最旧文件。"""
    backups = max(1, int(backups))
    oldest = p.with_name(f"{p.name}.{backups}")
    if oldest.exists():
        oldest.unlink()
    for i in range(backups - 1, 0, -1):
        src = p.with_name(f"{p.name}.{i}")
        if src.exists():
            src.replace(p.with_name(f"{p.name}.{i + 1}"))
    p.replace(p.with_name(f"{p.name}.1"))


def append_jsonl(path: str | Path, record: dict, max_bytes: int = 0, backups: int = 3) -> Path:
    """向 JSONL 追加一条记录；当前文件达到 max_bytes 时轮转并只保留 backups 个历史文件。

    为什么必须有轮转：审计台账（`audit_alerts.jsonl`）与标注回流（`review_feedback.jsonl`）
    都是**只追加、从不删除**的。长期运行会把磁盘写满，也会让"出事回查"变成一个打不开的巨型文件。
    `max_bytes<=0` 表示不轮转（一次性工具与测试用）。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if max_bytes and max_bytes > 0 and p.exists() and p.stat().st_size >= int(max_bytes):
        _rotate_jsonl(p, backups)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return p
