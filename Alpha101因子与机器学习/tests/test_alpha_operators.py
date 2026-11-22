"""Alpha101基础运算符单元测试。"""

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

from csi300_ml.alpha_operators import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    差分,
    带符号幂,
    延迟,
    时间序列均值,
    时间序列排名,
    时间序列最大值位置,
    横截面排名,
    横截面缩放,
)


class 基础运算符测试(unittest.TestCase):
    """验证窗口口径、并列排名和未来数据隔离。"""

    def setUp(self) -> None:
        self.样本 = pd.DataFrame(
            {
                "股票甲": [1.0, 2.0, 2.0, 4.0, 5.0],
                "股票乙": [5.0, 4.0, 3.0, 2.0, 1.0],
            },
            index=pd.date_range("2020-01-01", periods=5),
        )

    def test_延迟与差分(self) -> None:
        self.assertTrue(延迟(self.样本, 1).iloc[0].isna().all())
        self.assertEqual(延迟(self.样本, 1).iloc[1, 0], 1.0)
        self.assertEqual(差分(self.样本, 2).iloc[2, 1], -2.0)

    def test_窗口不足返回缺失(self) -> None:
        均值 = 时间序列均值(self.样本, 3)
        self.assertTrue(均值.iloc[:2].isna().all().all())
        self.assertAlmostEqual(均值.iloc[2, 0], 5 / 3)

    def test_时间序列并列排名(self) -> None:
        排名 = 时间序列排名(self.样本[["股票甲"]], 3)
        self.assertAlmostEqual(排名.iloc[2, 0], 2.5 / 3)

    def test_最大值位置从旧到新计数(self) -> None:
        位置 = 时间序列最大值位置(self.样本[["股票乙"]], 3)
        self.assertEqual(位置.iloc[2, 0], 1.0)

    def test_横截面运算(self) -> None:
        排名 = 横截面排名(self.样本)
        self.assertEqual(排名.iloc[0, 0], 0.5)
        self.assertEqual(排名.iloc[0, 1], 1.0)

        缩放 = 横截面缩放(self.样本)
        np.testing.assert_allclose(缩放.abs().sum(axis=1), 1.0)

    def test_带符号幂(self) -> None:
        数据 = pd.DataFrame({"数值": [-4.0, 0.0, 9.0]})
        结果 = 带符号幂(数据, 0.5)
        np.testing.assert_allclose(结果["数值"], [-2.0, 0.0, 3.0])

    def test_未来数据不改变历史结果(self) -> None:
        原结果 = 时间序列均值(self.样本, 3)
        修改后样本 = self.样本.copy()
        修改后样本.iloc[-1] = 999.0
        修改后结果 = 时间序列均值(修改后样本, 3)
        assert_frame_equal(原结果.iloc[:-1], 修改后结果.iloc[:-1])


if __name__ == "__main__":
    unittest.main()
