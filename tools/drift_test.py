"""分布漂移鲁棒性测试：换个"网络环境/攻击工具"还准不准？

做法
----
1. 用生成器在 `drift_strength>0` 下再生成一份**漂移测试集**
   （包规模、时长、包长整体缩放，相当于攻击工具换代/链路带宽变化）；
2. 用**训练时保存的 scaler** 做同样的预处理（绝不重新拟合），保证评估口径一致；
3. 对所有已训练模型跑一遍，输出「同分布 vs 漂移」对照表。

用法：
    python tools/drift_test.py                  # 默认强度 0.35
    python tools/drift_test.py --strength 0.5 --models cnn_bilstm_main_s0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data_pipeline import PreparedData  # noqa: E402
from src.evaluate import drift_evaluation  # noqa: E402
from src.model import load_model_bundle  # noqa: E402
from src.traffic_generator import generate_datasets  # noqa: E402
from src.utils import ARTIFACT_DIR, RAW_DIR, RESULT_DIR, ensure_dirs, get_logger, markdown_table, save_text, set_seed  # noqa: E402

LOGGER = get_logger("cida.drift")


def ensure_drift_set(cfg: Config, strength: float) -> Path:
    """确保漂移测试集存在（不存在就用同一个随机种子生成，主数据集内容不变）。"""
    path = RAW_DIR / f"{cfg.schema}_drift_test.csv"
    if path.exists():
        LOGGER.info("复用已有漂移测试集：%s", path.name)
        return path
    cfg_d = Config.from_dict(cfg.to_dict())
    cfg_d.drift_strength = float(strength)
    LOGGER.info("生成漂移测试集（drift_strength=%.2f）……", strength)
    files = generate_datasets(cfg_d)
    return files.get("drift", path)


def find_models(only: list[str] | None = None) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for d in sorted(ARTIFACT_DIR.iterdir() if ARTIFACT_DIR.exists() else []):
        if not d.is_dir() or (only and d.name not in only):
            continue
        for name in ("model.keras", "baseline.joblib"):
            p = d / name
            if p.exists():
                out.append((d.name, p))
                break
    return out


def load_any(path: Path):
    return joblib.load(path) if path.suffix == ".joblib" else load_model_bundle(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="分布漂移鲁棒性测试")
    ap.add_argument("--strength", type=float, default=0.35, help="漂移强度（对数正态标准差）")
    ap.add_argument("--models", nargs="*", default=None, help="只测这些 tag（默认全部）")
    args = ap.parse_args(argv)

    ensure_dirs()
    cfg = Config()
    set_seed(cfg.seed)
    data = PreparedData.load()
    drift_path = ensure_drift_set(cfg, args.strength)
    if not drift_path.exists():
        LOGGER.error("无法生成漂移测试集")
        return 1

    # 读一下主实验的"同分布"指标，用于对照
    base_reports: dict[str, dict] = {}
    for p in RESULT_DIR.glob("metrics_*.json"):
        try:
            base_reports[p.stem.replace("metrics_", "")] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass

    rows, detail = [], {}
    for tag, path in find_models(args.models):
        try:
            model = load_any(path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("跳过 %s：%s", tag, exc)
            continue
        rep = drift_evaluation(model, data, cfg, drift_path=drift_path)
        if rep is None:
            LOGGER.warning("跳过 %s：漂移评估不可用", tag)
            continue
        base = base_reports.get(tag, {})
        bm, dm = (base.get("multiclass") or {}), rep["multiclass"]
        bb, db = (base.get("binary") or {}), rep["binary"]
        detail[tag] = {"in_distribution": {"multiclass": bm, "binary": bb}, "drift": rep}
        rows.append([
            tag,
            f"{bm.get('macro_f1', float('nan')):.4f}" if bm else "-",
            f"{dm['macro_f1']:.4f}",
            f"{bb.get('f1', float('nan')):.4f}" if bb else "-",
            f"{db['f1']:.4f}",
            f"{bb.get('roc_auc', float('nan')):.4f}" if bb else "-",
            f"{db['roc_auc']:.4f}",
        ])
        LOGGER.info("[%s] 漂移后：宏F1 %.4f（同分布 %s）| 二分类 F1 %.4f | AUC %.4f",
                    tag, dm["macro_f1"],
                    f"{bm.get('macro_f1'):.4f}" if bm else "-", db["f1"], db["roc_auc"])

    if not rows:
        LOGGER.error("没有任何模型完成漂移评估")
        return 1

    table = markdown_table(
        ["模型", "宏F1（同分布）", "宏F1（漂移集）", "二分类F1（同分布）", "二分类F1（漂移集）",
         "AUC（同分布）", "AUC（漂移集）"],
        rows,
    )
    text = (
        f"# 分布漂移鲁棒性测试（drift_strength={args.strength}）\n\n"
        "漂移集由生成器按同一套攻击机理、对包规模/时长/包长做整体缩放后生成，"
        "模拟「攻击工具换代」或「链路带宽变化」。预处理沿用训练时的 scaler（不重新拟合）。\n\n"
        f"{table}\n\n"
        "**怎么读这张表**：同一列左右两数差距越小，说明模型对分布变化的鲁棒性越好。\n"
    )
    save_text(text, RESULT_DIR / "drift_report.md")
    (RESULT_DIR / "drift_report.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(text)
    LOGGER.info("漂移报告已保存 → %s", RESULT_DIR / "drift_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
