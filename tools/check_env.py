"""环境自检：一条命令告诉你"还缺什么、能不能跑"。

    python tools/check_env.py
"""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 先导入 src.utils：它会把 matplotlib/keras 的缓存目录收敛到项目内（MPLCONFIGDIR/KERAS_HOME），
# 否则在受限环境下导入 seaborn/matplotlib 会因为写不了用户主目录而报 PermissionError。
from src.utils import ARTIFACT_DIR, FIGURE_DIR, PROCESSED_DIR, RAW_DIR, RESULT_DIR  # noqa: E402

REQUIRED = {
    "numpy": "数值计算（必需）",
    "pandas": "数据表处理（必需）",
    "sklearn": "标准化/编码/指标（必需）",
    "matplotlib": "出图（必需）",
    "seaborn": "混淆矩阵热力图（必需）",
    "joblib": "保存 scaler/编码器（必需）",
    "tensorflow": "CNN+BiLSTM 训练与推理（训练必需，演示可不装）",
    "flask": "可视化演示大屏（演示必需）",
}
OPTIONAL = {"seaborn", "flask", "tensorflow"}


def main() -> int:
    print("=" * 72)
    print("环境自检")
    print("=" * 72)
    print(f"Python  : {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
    print(f"解释器  : {sys.executable}")
    print(f"项目根  : {ROOT}")
    print("-" * 72)

    missing_required, missing_optional = [], []
    for name, desc in REQUIRED.items():
        try:
            mod = importlib.import_module(name)
            print(f"[✓] {name:<14} {getattr(mod, '__version__', '?'):<12} {desc}")
        except Exception as exc:  # noqa: BLE001
            is_optional = name in OPTIONAL
            flag = "[ ]" if is_optional else "[×]"
            print(f"{flag} {name:<14} {'未安装':<12} {desc}  ({type(exc).__name__})")
            (missing_optional if is_optional else missing_required).append(name)

    print("-" * 72)
    # 产物检查
    checks = [
        (RAW_DIR, "原始仿真数据"),
        (PROCESSED_DIR / "tensors.npz", "预处理张量"),
        (ARTIFACT_DIR, "模型产物目录"),
        (RESULT_DIR, "指标结果目录"),
        (FIGURE_DIR, "图表目录"),
    ]
    for path, desc in checks:
        exists = path.exists() and (path.is_file() or any(path.iterdir()))
        print(f"{'[✓]' if exists else '[ ]'} {desc:<14} {path}")

    print("-" * 72)
    if missing_required:
        print("缺少必需依赖：", ", ".join(missing_required))
        print("安装： pip install -r requirements.txt")
        return 1
    if missing_optional:
        print("缺少可选依赖：", ", ".join(missing_optional))
        print("  * 缺 tensorflow → 无法训练，但可用 `python tools/smoke_test.py` 验证数据链路")
        print("  * 缺 flask     → 无法启动演示大屏，其余流程不受影响")
    else:
        print("全部依赖就绪 ✅")
    print("下一步： python run_all.py --smoke      # 快速跑通全流程")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
