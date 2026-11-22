"""第一阶段：审计历史成分股链条与行情覆盖率。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


项目根目录 = Path(__file__).resolve().parents[1]
源码目录 = 项目根目录 / "src"
if str(源码目录) not in sys.path:
    sys.path.insert(0, str(源码目录))

from csi300_ml.data_audit import 生成审计报告  # pyright: ignore[reportMissingImports]  # noqa: E402


初始成分快照 = 项目根目录 / "沪深300成分详情19 1 2.xlsx"
成分调整记录 = 项目根目录 / "沪深300成分进出记录(000300.SH).xlsx"
行情数据文件 = 项目根目录 / "沪深300选股数据导出19-25.xlsx"
输出目录 = 项目根目录 / "outputs" / "data_audit"


def 主程序() -> None:
    """运行审计并输出机器可读结果。"""
    必需文件 = (初始成分快照, 成分调整记录, 行情数据文件)
    缺失文件 = [文件.name for 文件 in 必需文件 if not 文件.exists()]
    if 缺失文件:
        raise FileNotFoundError(f"缺少输入文件: {缺失文件}")

    报告, _调整链条, _缺失代码, _额外代码, _已覆盖代码 = 生成审计报告(
        初始快照路径=初始成分快照,
        调整记录路径=成分调整记录,
        行情文件路径=行情数据文件,
    )

    输出目录.mkdir(parents=True, exist_ok=True)
    (输出目录 / "audit_summary.json").write_text(
        json.dumps(报告, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    覆盖情况 = 报告["覆盖情况"]
    成分股情况 = 报告["成分股"]
    print("数据审计完成")
    print(f"历史成分股并集: {成分股情况['历史并集数量']}只")
    print(f"行情已覆盖: {覆盖情况['已覆盖历史代码数']}只")
    print(f"行情缺失: {覆盖情况['缺失历史代码数']}只")
    print(f"行情额外代码: {覆盖情况['额外行情代码数']}只")
    print(f"覆盖率: {覆盖情况['覆盖率']:.2%}")
    print(f"成分调整链条有效: {成分股情况['调整链条有效']}")
    print(f"可进入完整动态研究: {覆盖情况['可开展完整动态研究']}")
    print(f"结果目录: {输出目录}")


if __name__ == "__main__":
    主程序()
