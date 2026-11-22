"""第一批Alpha101因子测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


项目根目录 = Path(__file__).resolve().parents[1]
源码目录 = 项目根目录 / "src"
if str(源码目录) not in sys.path:
    sys.path.insert(0, str(源码目录))

from csi300_ml.alpha_factors import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    第一批因子名称,
    第二批因子名称,
    计算第一批因子,
    计算第二批因子,
)


class 第一批因子测试(unittest.TestCase):
    """验证因子集合、公式和未来数据隔离。"""

    def setUp(self) -> None:
        日期 = pd.date_range("2020-01-01", periods=30)
        股票 = ["甲", "乙", "丙", "丁"]
        基准 = np.arange(30, dtype=float)[:, None]
        股票差异 = np.arange(4, dtype=float)[None, :]
        self.开盘价 = pd.DataFrame(
            10 + 基准 * 0.1 + 股票差异 * 0.3,
            index=日期,
            columns=股票,
        )
        self.收盘价 = self.开盘价 + np.sin(基准 + 股票差异) * 0.2
        self.最高价 = pd.DataFrame(
            np.maximum(self.开盘价, self.收盘价) + 0.3,
            index=日期,
            columns=股票,
        )
        self.最低价 = pd.DataFrame(
            np.minimum(self.开盘价, self.收盘价) - 0.3,
            index=日期,
            columns=股票,
        )
        self.成交量 = pd.DataFrame(
            1000 + 基准 * (10 + 股票差异) + 股票差异 * 50,
            index=日期,
            columns=股票,
        )
        self.掩码 = pd.DataFrame(True, index=日期, columns=股票)

    def _计算(self) -> dict[str, pd.DataFrame]:
        return 计算第一批因子(
            self.开盘价,
            self.收盘价,
            self.最高价,
            self.最低价,
            self.成交量,
            self.掩码,
        )

    def _计算第二批(self) -> dict[str, pd.DataFrame]:
        return 计算第二批因子(
            self.开盘价,
            self.收盘价,
            self.最高价,
            self.最低价,
            self.成交量,
            self.掩码,
        )

    def test_因子名称和维度完整(self) -> None:
        结果 = self._计算()
        self.assertEqual(set(结果), set(第一批因子名称))
        for 因子 in 结果.values():
            self.assertEqual(因子.shape, self.开盘价.shape)

    def test_Alpha12公式(self) -> None:
        结果 = self._计算()["Alpha12"]
        日期位置, 股票位置 = 10, 2
        成交量变化 = (
            self.成交量.iloc[日期位置, 股票位置]
            - self.成交量.iloc[日期位置 - 1, 股票位置]
        )
        收盘价变化 = (
            self.收盘价.iloc[日期位置, 股票位置]
            - self.收盘价.iloc[日期位置 - 1, 股票位置]
        )
        期望值 = np.sign(成交量变化) * (-收盘价变化)
        self.assertAlmostEqual(结果.iloc[日期位置, 股票位置], 期望值)

    def test_未来扰动不改变历史因子(self) -> None:
        原结果 = {**self._计算(), **self._计算第二批()}
        原始输入 = (
            self.开盘价.copy(),
            self.收盘价.copy(),
            self.最高价.copy(),
            self.最低价.copy(),
            self.成交量.copy(),
        )
        self.开盘价.iloc[-1] = 999.0
        self.收盘价.iloc[-1] = 998.0
        self.最高价.iloc[-1] = 1000.0
        self.最低价.iloc[-1] = 997.0
        self.成交量.iloc[-1] = 999999.0
        修改后结果 = {**self._计算(), **self._计算第二批()}
        (
            self.开盘价,
            self.收盘价,
            self.最高价,
            self.最低价,
            self.成交量,
        ) = 原始输入

        for 因子名称 in 第一批因子名称 + 第二批因子名称:
            assert_frame_equal(
                原结果[因子名称].iloc[:-1],
                修改后结果[因子名称].iloc[:-1],
            )

    def test_第二批因子名称和维度完整(self) -> None:
        结果 = self._计算第二批()
        self.assertEqual(set(结果), set(第二批因子名称))
        for 因子 in 结果.values():
            self.assertEqual(因子.shape, self.开盘价.shape)


if __name__ == "__main__":
    unittest.main()
