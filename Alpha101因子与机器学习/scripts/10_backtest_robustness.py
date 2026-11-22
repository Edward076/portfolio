"""模型分组收益、非重叠组合、交易成本和稳健性检验。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


项目根目录 = Path(__file__).resolve().parents[1]
预测文件 = 项目根目录 / "outputs" / "models" / "model_predictions.parquet"
模型数据文件 = 项目根目录 / "data" / "processed" / "model_dataset.parquet"
结果目录 = 项目根目录 / "outputs" / "backtest"
结果工作簿 = 结果目录 / "backtest_robustness.xlsx"
组合明细文件 = 结果目录 / "weekly_portfolio.parquet"


def _计算每日分组(预测: pd.DataFrame) -> pd.DataFrame:
    """计算每个模型每日Top/Bottom 20%收益。"""
    模型列 = [列 for 列 in 预测.columns if 列.endswith("_预测概率")]
    记录: list[dict[str, object]] = []
    for 日期, 当日数据 in 预测.groupby("日期", sort=True):
        for 模型列名 in 模型列:
            分位排名 = 当日数据[模型列名].rank(pct=True, method="average")
            最高组 = 当日数据.loc[分位排名.ge(0.8)]
            最低组 = 当日数据.loc[分位排名.le(0.2)]
            记录.append(
                {
                    "日期": 日期,
                    "模型": 模型列名.removesuffix("_预测概率"),
                    "Top20收益": 最高组["未来5日收益"].mean(),
                    "Bottom20收益": 最低组["未来5日收益"].mean(),
                    "Top-Bottom": (
                        最高组["未来5日收益"].mean()
                        - 最低组["未来5日收益"].mean()
                    ),
                    "Top20相对收益": 最高组["未来5日相对收益"].mean(),
                }
            )
    return pd.DataFrame(记录)


def _计算换手率(
    当前权重: dict[str, float],
    上期权重: dict[str, float],
) -> float:
    """根据目标权重变化计算单边换手率。"""
    全部代码 = set(当前权重).union(上期权重)
    return 0.5 * sum(
        abs(当前权重.get(代码, 0.0) - 上期权重.get(代码, 0.0))
        for 代码 in 全部代码
    )


def _构造非重叠组合(
    测试预测: pd.DataFrame,
    模型列: str,
) -> pd.DataFrame:
    """每5个交易日选取一次Top 20%等权组合。"""
    全部日期 = sorted(pd.to_datetime(测试预测["日期"]).unique())
    调仓日期 = set(全部日期[::5])
    上期权重: dict[str, float] = {}
    记录: list[dict[str, object]] = []

    for 日期, 当日数据 in 测试预测.groupby("日期", sort=True):
        if pd.Timestamp(日期) not in 调仓日期:
            continue
        分位排名 = 当日数据[模型列].rank(pct=True, method="average")
        持仓 = 当日数据.loc[分位排名.ge(0.8)].copy()
        if 持仓.empty:
            continue
        权重 = 1 / len(持仓)
        当前权重 = dict.fromkeys(持仓["证券代码"], 权重)
        换手率 = _计算换手率(当前权重, 上期权重)
        记录.append(
            {
                "日期": 日期,
                "持仓数量": len(持仓),
                "换手率": 换手率,
                "组合收益": 持仓["未来5日收益"].mean(),
                "组合相对收益": 持仓["未来5日相对收益"].mean(),
            }
        )
        上期权重 = 当前权重
    return pd.DataFrame(记录)


def _绩效指标(
    收益序列: pd.Series,
    年化期数: float = 50.4,
) -> dict[str, float]:
    """计算非重叠5日收益序列的常见绩效指标。"""
    收益 = 收益序列.dropna()
    净值 = (1 + 收益).cumprod()
    回撤 = 净值 / 净值.cummax() - 1
    波动率 = 收益.std(ddof=1)
    return {
        "累计收益": float(净值.iloc[-1] - 1),
        "年化收益": float((净值.iloc[-1] ** (年化期数 / len(收益))) - 1),
        "年化波动率": float(波动率 * np.sqrt(年化期数)),
        "夏普比率": float(
            收益.mean() / 波动率 * np.sqrt(年化期数)
            if 波动率 > 0
            else np.nan
        ),
        "最大回撤": float(回撤.min()),
        "胜率": float((收益 > 0).mean()),
        "期数": len(收益),
    }


def _计算预测周期IC(
    数据: pd.DataFrame,
    概率列: str,
) -> pd.DataFrame:
    """检验同一模型分数对1日、5日、10日收益的Rank IC。"""
    记录 = []
    for 周期 in (1, 5, 10):
        收益列 = f"未来{周期}日相对收益"
        每日IC = 数据.groupby("日期").apply(
            lambda 当日: 当日[概率列].corr(
                当日[收益列],
                method="spearman",
            ),
            include_groups=False,
        ).dropna()
        记录.append(
            {
                "预测周期": 周期,
                "RankIC均值": 每日IC.mean(),
                "正IC比例": (每日IC > 0).mean(),
                "有效交易日": len(每日IC),
            }
        )
    return pd.DataFrame(记录)


def 主程序() -> None:
    """完成金融评价、成本和主要稳健性检验。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    预测 = pd.read_parquet(预测文件)
    预测["日期"] = pd.to_datetime(预测["日期"]).dt.normalize()
    测试预测 = 预测.loc[预测["数据集"] == "测试集"].copy()
    测试预测["年份"] = 测试预测["日期"].dt.year

    每日分组 = _计算每日分组(测试预测)
    模型分组汇总 = 每日分组.groupby("模型")[
        ["Top20收益", "Bottom20收益", "Top-Bottom", "Top20相对收益"]
    ].mean().reset_index()
    年度分组汇总 = 每日分组.assign(
        年份=pd.to_datetime(每日分组["日期"]).dt.year
    ).groupby(["年份", "模型"])[
        ["Top-Bottom", "Top20相对收益"]
    ].mean().reset_index()

    主模型列 = "完整LightGBM_预测概率"
    周度组合 = _构造非重叠组合(测试预测, 主模型列)
    成本记录 = []
    for 双边成本基点 in (0, 10, 20, 30):
        成本率 = 双边成本基点 / 10000
        净收益列 = (
            周度组合["组合相对收益"]
            - 周度组合["换手率"] * 成本率
        )
        成本记录.append(
            {
                "双边成本基点": 双边成本基点,
                **_绩效指标(净收益列),
            }
        )
    成本敏感性 = pd.DataFrame(成本记录)

    周度组合["毛相对净值"] = (1 + 周度组合["组合相对收益"]).cumprod()
    周度组合["20bps净相对收益"] = (
        周度组合["组合相对收益"] - 周度组合["换手率"] * 0.002
    )
    周度组合["20bps净相对净值"] = (
        1 + 周度组合["20bps净相对收益"]
    ).cumprod()

    模型数据 = pd.read_parquet(模型数据文件)
    测试标签 = 模型数据.loc[模型数据["数据集"] == "测试集", [
        "日期",
        "证券代码",
        "未来1日相对收益",
        "未来10日相对收益",
    ]]
    稳健数据 = 测试预测.merge(
        测试标签,
        on=["日期", "证券代码"],
        how="left",
        validate="one_to_one",
    )
    周期稳健性 = _计算预测周期IC(稳健数据, 主模型列)

    年度分类记录 = []
    模型列 = [列 for 列 in 测试预测.columns if 列.endswith("_预测概率")]
    for 年份, 年度数据 in 测试预测.groupby("年份"):
        for 概率列 in 模型列:
            年度分类记录.append(
                {
                    "年份": int(年份),
                    "模型": 概率列.removesuffix("_预测概率"),
                    "ROC_AUC": roc_auc_score(
                        年度数据["未来5日分类标签"],
                        年度数据[概率列],
                    ),
                }
            )
    年度分类 = pd.DataFrame(年度分类记录)

    结果目录.mkdir(parents=True, exist_ok=True)
    周度组合.to_parquet(组合明细文件, index=False, compression="zstd")
    with pd.ExcelWriter(结果工作簿, engine="xlsxwriter") as 写入器:
        模型分组汇总.to_excel(写入器, sheet_name="模型分组收益", index=False)
        年度分组汇总.to_excel(写入器, sheet_name="分年分组收益", index=False)
        成本敏感性.to_excel(写入器, sheet_name="交易成本敏感性", index=False)
        周期稳健性.to_excel(写入器, sheet_name="预测周期稳健性", index=False)
        年度分类.to_excel(写入器, sheet_name="分年分类指标", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(
        周度组合["日期"],
        周度组合["毛相对净值"],
        label="成本前",
    )
    plt.plot(
        周度组合["日期"],
        周度组合["20bps净相对净值"],
        label="双边成本20bps后",
    )
    plt.title("完整LightGBM Top20%非重叠组合相对净值")
    plt.xlabel("调仓日期")
    plt.ylabel("累计相对净值")
    plt.legend()
    plt.tight_layout()
    plt.savefig(结果目录 / "weekly_relative_nav.png", dpi=180)
    plt.close()

    排序汇总 = 模型分组汇总.sort_values("Top-Bottom")
    plt.figure(figsize=(9, 5))
    plt.barh(排序汇总["模型"], 排序汇总["Top-Bottom"])
    plt.title("各模型测试集Top-Bottom五日平均收益")
    plt.xlabel("五日平均收益差")
    plt.ylabel("模型")
    plt.tight_layout()
    plt.savefig(结果目录 / "model_top_bottom.png", dpi=180)
    plt.close()

    print("组合回测和稳健性检验完成")
    print(模型分组汇总.sort_values("Top-Bottom", ascending=False))
    print(成本敏感性)
    print(f"结果文件: {结果工作簿}")


if __name__ == "__main__":
    主程序()
