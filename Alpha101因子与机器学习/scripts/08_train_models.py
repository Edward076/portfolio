"""训练Logistic、Random Forest和LightGBM并进行样本外评价。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd


项目根目录 = Path(__file__).resolve().parents[1]
源码目录 = 项目根目录 / "src"
if str(源码目录) not in sys.path:
    sys.path.insert(0, str(源码目录))

from csi300_ml.alpha_factors import 全部因子名称  # pyright: ignore[reportMissingImports]  # noqa: E402
from csi300_ml.dataset_builder import 普通量价特征名称  # pyright: ignore[reportMissingImports]  # noqa: E402
from csi300_ml.modeling import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    创建LightGBM,
    创建逻辑回归,
    创建随机森林,
    评价模型,
)


模型数据文件 = 项目根目录 / "data" / "processed" / "model_dataset.parquet"
模型目录 = 项目根目录 / "models"
结果目录 = 项目根目录 / "outputs" / "models"
预测文件 = 结果目录 / "model_predictions.parquet"
指标文件 = 结果目录 / "model_metrics.xlsx"
配置文件 = 结果目录 / "best_model_config.json"


def 主程序() -> None:
    """在训练集拟合、验证集选型，并一次性评价测试集。"""
    数据 = pd.read_parquet(模型数据文件)
    训练数据 = 数据.loc[数据["数据集"] == "训练集"].copy()
    验证数据 = 数据.loc[数据["数据集"] == "验证集"].copy()
    测试数据 = 数据.loc[数据["数据集"] == "测试集"].copy()
    标签列 = "未来5日分类标签"
    收益列 = "未来5日相对收益"

    Alpha特征 = list(全部因子名称)
    量价特征 = list(普通量价特征名称)
    完整特征 = Alpha特征 + 量价特征
    模型方案 = {
        "量价Logistic": {
            "特征": 量价特征,
            "候选": [
                ("C=0.1", 创建逻辑回归(0.1)),
                ("C=1.0", 创建逻辑回归(1.0)),
            ],
        },
        "AlphaLogistic": {
            "特征": Alpha特征,
            "候选": [
                ("C=0.1", 创建逻辑回归(0.1)),
                ("C=1.0", 创建逻辑回归(1.0)),
            ],
        },
        "完整Logistic": {
            "特征": 完整特征,
            "候选": [
                ("C=0.1", 创建逻辑回归(0.1)),
                ("C=1.0", 创建逻辑回归(1.0)),
            ],
        },
        "完整RandomForest": {
            "特征": 完整特征,
            "候选": [
                ("最大深度=8", 创建随机森林(8)),
                ("最大深度=12", 创建随机森林(12)),
            ],
        },
        "AlphaLightGBM": {
            "特征": Alpha特征,
            "候选": [
                ("叶子数=15", 创建LightGBM(15)),
                ("叶子数=31", 创建LightGBM(31)),
            ],
        },
        "完整LightGBM": {
            "特征": 完整特征,
            "候选": [
                ("叶子数=15", 创建LightGBM(15)),
                ("叶子数=31", 创建LightGBM(31)),
            ],
        },
    }

    验证记录: list[dict[str, object]] = []
    最佳模型: dict[str, object] = {}
    最佳配置: dict[str, dict[str, object]] = {}
    验证预测 = 验证数据[
        ["日期", "证券代码", "证券名称", 标签列, 收益列, "未来5日收益"]
    ].copy()

    for 方案名称, 方案 in 模型方案.items():
        特征 = 方案["特征"]
        最佳AUC = -1.0
        for 参数名称, 模型 in 方案["候选"]:
            模型.fit(训练数据[特征], 训练数据[标签列])
            指标, 概率 = 评价模型(
                模型,
                验证数据[特征],
                验证数据[标签列],
                验证数据["日期"],
                验证数据[收益列],
            )
            验证记录.append(
                {
                    "模型": 方案名称,
                    "参数": 参数名称,
                    "数据集": "验证集",
                    **指标,
                }
            )
            if 指标["ROC_AUC"] > 最佳AUC:
                最佳AUC = 指标["ROC_AUC"]
                最佳模型[方案名称] = 模型
                最佳配置[方案名称] = {
                    "参数": 参数名称,
                    "特征": 特征,
                    "验证ROC_AUC": 最佳AUC,
                }
                验证预测[f"{方案名称}_预测概率"] = 概率

    测试记录: list[dict[str, object]] = []
    测试预测 = 测试数据[
        ["日期", "证券代码", "证券名称", 标签列, 收益列, "未来5日收益"]
    ].copy()
    模型目录.mkdir(parents=True, exist_ok=True)

    for 方案名称, 模型 in 最佳模型.items():
        特征 = 最佳配置[方案名称]["特征"]
        指标, 概率 = 评价模型(
            模型,
            测试数据[特征],
            测试数据[标签列],
            测试数据["日期"],
            测试数据[收益列],
        )
        测试记录.append(
            {
                "模型": 方案名称,
                "参数": 最佳配置[方案名称]["参数"],
                "数据集": "测试集",
                **指标,
            }
        )
        测试预测[f"{方案名称}_预测概率"] = 概率
        joblib.dump(
            {
                "模型": 模型,
                "特征": 特征,
                "配置": 最佳配置[方案名称],
            },
            模型目录 / f"{方案名称}.joblib",
        )

    验证预测["数据集"] = "验证集"
    测试预测["数据集"] = "测试集"
    全部预测 = pd.concat([验证预测, 测试预测], ignore_index=True)
    全部指标 = pd.DataFrame(验证记录 + 测试记录)
    测试指标 = pd.DataFrame(测试记录).sort_values(
        "ROC_AUC",
        ascending=False,
    )

    结果目录.mkdir(parents=True, exist_ok=True)
    全部预测.to_parquet(预测文件, index=False, compression="zstd")
    with pd.ExcelWriter(指标文件, engine="xlsxwriter") as 写入器:
        测试指标.to_excel(写入器, sheet_name="测试集模型比较", index=False)
        全部指标.to_excel(写入器, sheet_name="验证与测试指标", index=False)
        for 工作表 in 写入器.sheets.values():
            工作表.freeze_panes(1, 0)
            工作表.autofilter(0, 0, 工作表.dim_rowmax, 工作表.dim_colmax)
            工作表.set_column(0, 工作表.dim_colmax, 18)
    配置文件.write_text(
        json.dumps(最佳配置, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("模型训练和测试集评价完成")
    print(测试指标[["模型", "ROC_AUC", "每日等权AUC", "预测RankIC均值"]])
    print(f"指标文件: {指标文件}")


if __name__ == "__main__":
    主程序()
