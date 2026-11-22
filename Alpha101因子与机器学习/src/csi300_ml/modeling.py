"""机器学习模型训练、分类指标和金融预测指标。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def 创建逻辑回归(正则倒数: float) -> Pipeline:
    """创建带训练期中位数填补和标准化的逻辑回归。"""
    return Pipeline(
        [
            ("缺失填补", SimpleImputer(strategy="median")),
            ("标准化", StandardScaler()),
            (
                "模型",
                LogisticRegression(
                    C=正则倒数,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def 创建随机森林(最大深度: int) -> Pipeline:
    """创建控制复杂度的随机森林模型。"""
    return Pipeline(
        [
            ("缺失填补", SimpleImputer(strategy="median")),
            (
                "模型",
                RandomForestClassifier(
                    n_estimators=240,
                    max_depth=最大深度,
                    min_samples_leaf=80,
                    max_features="sqrt",
                    max_samples=0.75,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def 创建LightGBM(叶子数: int) -> Pipeline:
    """创建适用于表格型因子数据的LightGBM模型。"""
    return Pipeline(
        [
            ("缺失填补", SimpleImputer(strategy="median")),
            (
                "模型",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=360,
                    learning_rate=0.04,
                    num_leaves=叶子数,
                    max_depth=-1,
                    min_child_samples=100,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    n_jobs=-1,
                    random_state=42,
                    verbosity=-1,
                ),
            ),
        ]
    )


def 计算分类指标(
    真实标签: pd.Series,
    预测概率: np.ndarray,
) -> dict[str, float]:
    """计算分类概率和0.5阈值指标。"""
    预测类别 = (预测概率 >= 0.5).astype(int)
    return {
        "ROC_AUC": float(roc_auc_score(真实标签, 预测概率)),
        "PR_AUC": float(average_precision_score(真实标签, 预测概率)),
        "LogLoss": float(log_loss(真实标签, 预测概率)),
        "Brier": float(brier_score_loss(真实标签, 预测概率)),
        "Accuracy": float(accuracy_score(真实标签, 预测类别)),
        "Precision": float(
            precision_score(真实标签, 预测类别, zero_division=0)
        ),
        "Recall": float(recall_score(真实标签, 预测类别, zero_division=0)),
        "F1": float(f1_score(真实标签, 预测类别, zero_division=0)),
    }


def 计算每日等权AUC(
    日期: pd.Series,
    真实标签: pd.Series,
    预测概率: np.ndarray,
) -> float:
    """先逐日计算AUC，再对交易日等权平均。"""
    临时表 = pd.DataFrame(
        {
            "日期": pd.to_datetime(日期).to_numpy(),
            "真实标签": 真实标签.to_numpy(),
            "预测概率": 预测概率,
        }
    )
    每日AUC: list[float] = []
    for _, 当日数据 in 临时表.groupby("日期"):
        if 当日数据["真实标签"].nunique() < 2:
            continue
        每日AUC.append(
            roc_auc_score(当日数据["真实标签"], 当日数据["预测概率"])
        )
    return float(np.mean(每日AUC)) if 每日AUC else np.nan


def 计算预测RankIC(
    日期: pd.Series,
    相对收益: pd.Series,
    预测概率: np.ndarray,
) -> tuple[float, float]:
    """计算逐日预测概率Rank IC的均值和正IC比例。"""
    临时表 = pd.DataFrame(
        {
            "日期": pd.to_datetime(日期).to_numpy(),
            "相对收益": 相对收益.to_numpy(),
            "预测概率": 预测概率,
        }
    )
    每日IC = 临时表.groupby("日期").apply(
        lambda 当日: 当日["预测概率"].corr(
            当日["相对收益"],
            method="spearman",
        ),
        include_groups=False,
    ).dropna()
    return float(每日IC.mean()), float((每日IC > 0).mean())


def 评价模型(
    模型: BaseEstimator,
    特征: pd.DataFrame,
    标签: pd.Series,
    日期: pd.Series,
    相对收益: pd.Series,
) -> tuple[dict[str, float], np.ndarray]:
    """生成概率预测并同时计算机器学习和金融指标。"""
    预测概率 = 模型.predict_proba(特征)[:, 1]
    指标 = 计算分类指标(标签, 预测概率)
    指标["每日等权AUC"] = 计算每日等权AUC(日期, 标签, 预测概率)
    IC均值, 正IC比例 = 计算预测RankIC(日期, 相对收益, 预测概率)
    指标["预测RankIC均值"] = IC均值
    指标["预测正IC比例"] = 正IC比例
    return 指标, 预测概率


def 转换为普通字典(参数: dict[str, Any]) -> dict[str, Any]:
    """将NumPy标量转换为可JSON序列化的Python标量。"""
    return {
        键: 值.item() if isinstance(值, np.generic) else 值
        for 键, 值 in 参数.items()
    }
