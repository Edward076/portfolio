"""使用SHAP和置换重要性解释完整LightGBM模型。"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance


项目根目录 = Path(__file__).resolve().parents[1]
源码目录 = 项目根目录 / "src"
if str(源码目录) not in sys.path:
    sys.path.insert(0, str(源码目录))

模型数据文件 = 项目根目录 / "data" / "processed" / "model_dataset.parquet"
模型文件 = 项目根目录 / "models" / "完整LightGBM.joblib"
结果目录 = 项目根目录 / "outputs" / "shap"


def _提取SHAP数组(
    解释器: shap.TreeExplainer,
    特征值: np.ndarray,
) -> np.ndarray:
    """兼容不同SHAP版本的二分类返回格式。"""
    SHAP值 = 解释器.shap_values(特征值)
    if isinstance(SHAP值, list):
        return np.asarray(SHAP值[-1])
    数组 = np.asarray(SHAP值)
    if 数组.ndim == 3:
        return 数组[:, :, -1]
    return 数组


def 主程序() -> None:
    """生成全局、分年度SHAP和置换重要性结果。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    模型包 = joblib.load(模型文件)
    模型 = 模型包["模型"]
    特征名称: list[str] = 模型包["特征"]
    数据 = pd.read_parquet(模型数据文件)
    测试数据 = 数据.loc[数据["数据集"] == "测试集"].copy()
    测试数据["年份"] = pd.to_datetime(测试数据["日期"]).dt.year

    各年样本 = []
    for _, 年度数据 in 测试数据.groupby("年份"):
        各年样本.append(
            年度数据.sample(
                n=min(3000, len(年度数据)),
                random_state=42,
            )
        )
    解释样本 = pd.concat(各年样本, ignore_index=True)

    缺失填补器 = 模型.named_steps["缺失填补"]
    树模型 = 模型.named_steps["模型"]
    处理后特征 = 缺失填补器.transform(解释样本[特征名称])
    解释器 = shap.TreeExplainer(树模型)
    SHAP值 = _提取SHAP数组(解释器, 处理后特征)

    全局重要性 = pd.DataFrame(
        {
            "特征": 特征名称,
            "平均绝对SHAP": np.abs(SHAP值).mean(axis=0),
        }
    ).sort_values("平均绝对SHAP", ascending=False)

    年度重要性记录: list[pd.DataFrame] = []
    for 年份 in sorted(解释样本["年份"].unique()):
        年份掩码 = 解释样本["年份"].eq(年份).to_numpy()
        年度表 = pd.DataFrame(
            {
                "年份": int(年份),
                "特征": 特征名称,
                "平均绝对SHAP": np.abs(SHAP值[年份掩码]).mean(axis=0),
            }
        )
        年度重要性记录.append(年度表)
    年度重要性 = pd.concat(年度重要性记录, ignore_index=True)

    置换样本 = 测试数据.sample(
        n=min(5000, len(测试数据)),
        random_state=42,
    )
    置换结果 = permutation_importance(
        模型,
        置换样本[特征名称],
        置换样本["未来5日分类标签"],
        scoring="roc_auc",
        n_repeats=3,
        random_state=42,
        n_jobs=-1,
    )
    置换重要性 = pd.DataFrame(
        {
            "特征": 特征名称,
            "AUC下降均值": 置换结果.importances_mean,
            "AUC下降标准差": 置换结果.importances_std,
        }
    ).sort_values("AUC下降均值", ascending=False)

    结果目录.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(
        结果目录 / "feature_importance.xlsx",
        engine="xlsxwriter",
    ) as 写入器:
        全局重要性.to_excel(写入器, sheet_name="全局SHAP", index=False)
        年度重要性.to_excel(写入器, sheet_name="年度SHAP", index=False)
        置换重要性.to_excel(写入器, sheet_name="置换重要性", index=False)

    shap.summary_plot(
        SHAP值,
        处理后特征,
        feature_names=特征名称,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(结果目录 / "shap_importance_bar.png", dpi=180, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        SHAP值,
        处理后特征,
        feature_names=特征名称,
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(结果目录 / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    for 特征 in 全局重要性.head(3)["特征"]:
        shap.dependence_plot(
            特征,
            SHAP值,
            处理后特征,
            feature_names=特征名称,
            interaction_index=None,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            结果目录 / f"shap_dependence_{特征}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close()

    print("SHAP和置换重要性分析完成")
    print(全局重要性.head(10))
    print(f"结果目录: {结果目录}")


if __name__ == "__main__":
    主程序()
