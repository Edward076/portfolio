"""根据实证结果自动生成正式Word研究报告。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


项目根目录 = Path(__file__).resolve().parents[1]
报告目录 = 项目根目录 / "reports"
图表目录 = 项目根目录 / "outputs" / "report_figures"
报告文件 = 报告目录 / "正式研究报告.docx"


def _设置单元格文本(单元格, 文本: object) -> None:
    """设置表格单元格文字和中文字体。"""
    单元格.text = str(文本)
    for 段落 in 单元格.paragraphs:
        for 文字块 in 段落.runs:
            文字块.font.name = "宋体"
            文字块._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
            文字块.font.size = Pt(9)


def _添加数据表(
    文档: Document,
    数据: pd.DataFrame,
    显示列: list[str],
    小数位数: int = 4,
) -> None:
    """向Word添加格式化结果表。"""
    表格 = 文档.add_table(rows=1, cols=len(显示列))
    表格.style = "Table Grid"
    for 列号, 列名 in enumerate(显示列):
        _设置单元格文本(表格.rows[0].cells[列号], 列名)
    for _, 数据行 in 数据[显示列].iterrows():
        单元格 = 表格.add_row().cells
        for 列号, 列名 in enumerate(显示列):
            值 = 数据行[列名]
            if isinstance(值, float):
                文本 = f"{值:.{小数位数}f}"
            else:
                文本 = 值
            _设置单元格文本(单元格[列号], 文本)
    文档.add_paragraph()


def _添加正文(文档: Document, 文本: str) -> None:
    """添加首行缩进的中文正文。"""
    段落 = 文档.add_paragraph()
    段落.paragraph_format.first_line_indent = Cm(0.74)
    段落.paragraph_format.line_spacing = 1.5
    段落.paragraph_format.space_after = Pt(6)
    文字块 = 段落.add_run(文本)
    文字块.font.name = "宋体"
    文字块._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    文字块.font.size = Pt(11)


def _添加图片(
    文档: Document,
    图片路径: Path,
    标题: str,
    宽度: float = 15.5,
) -> None:
    """添加居中图片和图注。"""
    if not 图片路径.exists():
        return
    段落 = 文档.add_paragraph()
    段落.alignment = WD_ALIGN_PARAGRAPH.CENTER
    段落.add_run().add_picture(str(图片路径), width=Cm(宽度))
    图注 = 文档.add_paragraph(标题)
    图注.alignment = WD_ALIGN_PARAGRAPH.CENTER
    图注.runs[0].font.name = "宋体"
    图注.runs[0]._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    图注.runs[0].font.size = Pt(9)


def _添加页码(节) -> None:
    """在页脚添加居中页码域。"""
    页脚段落 = 节.footer.paragraphs[0]
    页脚段落.alignment = WD_ALIGN_PARAGRAPH.CENTER
    文字块 = 页脚段落.add_run()
    开始域 = OxmlElement("w:fldChar")
    开始域.set(qn("w:fldCharType"), "begin")
    指令 = OxmlElement("w:instrText")
    指令.set(qn("xml:space"), "preserve")
    指令.text = "PAGE"
    结束域 = OxmlElement("w:fldChar")
    结束域.set(qn("w:fldCharType"), "end")
    文字块._r.extend([开始域, 指令, 结束域])


def _生成补充图表(
    模型结果: pd.DataFrame,
    因子结果: pd.DataFrame,
) -> None:
    """生成报告所需的模型和因子比较图。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    图表目录.mkdir(parents=True, exist_ok=True)

    模型排序 = 模型结果.sort_values("ROC_AUC")
    plt.figure(figsize=(9, 5))
    plt.barh(模型排序["模型"], 模型排序["ROC_AUC"])
    plt.axvline(0.5, color="black", linewidth=1, linestyle="--")
    plt.title("2024—2025年测试集模型ROC-AUC")
    plt.xlabel("ROC-AUC")
    plt.ylabel("模型")
    plt.tight_layout()
    plt.savefig(图表目录 / "model_auc.png", dpi=180)
    plt.close()

    因子排序 = 因子结果.sort_values("IC均值").tail(12)
    plt.figure(figsize=(9, 6))
    plt.barh(因子排序["因子"], 因子排序["IC均值"])
    plt.title("测试集主要Alpha101因子Rank IC均值")
    plt.xlabel("Rank IC均值")
    plt.ylabel("因子")
    plt.tight_layout()
    plt.savefig(图表目录 / "factor_ic.png", dpi=180)
    plt.close()


def 主程序() -> None:
    """读取全部结果并生成正式研究报告。"""
    因子结果 = pd.read_excel(
        项目根目录 / "outputs/factor_evaluation/factor_evaluation.xlsx",
        sheet_name="测试集总览",
    )
    模型结果 = pd.read_excel(
        项目根目录 / "outputs/models/model_metrics.xlsx",
        sheet_name="测试集模型比较",
    )
    SHAP结果 = pd.read_excel(
        项目根目录 / "outputs/shap/feature_importance.xlsx",
        sheet_name="全局SHAP",
    )
    成本结果 = pd.read_excel(
        项目根目录 / "outputs/backtest/backtest_robustness.xlsx",
        sheet_name="交易成本敏感性",
    )
    周期结果 = pd.read_excel(
        项目根目录 / "outputs/backtest/backtest_robustness.xlsx",
        sheet_name="预测周期稳健性",
    )
    分组结果 = pd.read_excel(
        项目根目录 / "outputs/backtest/backtest_robustness.xlsx",
        sheet_name="模型分组收益",
    )
    _生成补充图表(模型结果, 因子结果)

    文档 = Document()
    节 = 文档.sections[0]
    节.top_margin = Cm(2.5)
    节.bottom_margin = Cm(2.5)
    节.left_margin = Cm(2.7)
    节.right_margin = Cm(2.7)
    _添加页码(节)

    标题 = 文档.add_paragraph()
    标题.alignment = WD_ALIGN_PARAGRAPH.CENTER
    标题.paragraph_format.space_before = Pt(80)
    标题文字 = 标题.add_run("基于精选Alpha101因子与机器学习的\n沪深300股票收益预测")
    标题文字.bold = True
    标题文字.font.size = Pt(22)
    标题文字.font.name = "黑体"
    标题文字._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")

    英文标题 = 文档.add_paragraph(
        "Cross-Sectional Return Prediction for CSI 300 Stocks\n"
        "Using Selected Alpha101 Factors and Machine Learning"
    )
    英文标题.alignment = WD_ALIGN_PARAGRAPH.CENTER
    英文标题.runs[0].font.size = Pt(13)
    文档.add_paragraph("\n\n")
    日期段落 = 文档.add_paragraph("研究区间：2019—2025年")
    日期段落.alignment = WD_ALIGN_PARAGRAPH.CENTER
    日期段落.runs[0].font.size = Pt(12)
    限制段落 = 文档.add_paragraph(
        "说明：本报告基于可得历史成分子样本，结论属于探索性研究。"
    )
    限制段落.alignment = WD_ALIGN_PARAGRAPH.CENTER
    限制段落.runs[0].font.size = Pt(11)
    文档.add_page_break()

    文档.add_heading("摘要", level=1)
    _添加正文(
        文档,
        "本文使用2019—2025年沪深300相关股票的前复权OHLC、成交量与成交金额数据，"
        "复现18个不依赖VWAP、行业中性化和市值字段的Alpha101因子，并研究其对未来5个"
        "完整交易日横截面相对收益的预测能力。信号在t日收盘后生成，t+1日开盘建仓，"
        "t+6日开盘退出。模型包括Logistic Regression、Random Forest和LightGBM，"
        "训练期为2019—2022年，验证期为2023年，测试期为2024—2025年。",
    )
    _添加正文(
        文档,
        "测试结果显示，完整Random Forest和完整LightGBM的ROC-AUC分别为0.5158和"
        "0.5153，预测Rank IC均值分别为0.0308和0.0295。Alpha Logistic的AUC为0.5138，"
        "普通量价Logistic仅为0.5013，说明当前样本中的预测增量主要来自Alpha因子。"
        "Alpha8、Alpha6、Alpha18和Alpha16具有较稳定的单因子IC。SHAP结果表明，20日"
        "收益率、20日波动率、Alpha6、成交金额对数和Alpha13是完整LightGBM的重要特征。",
    )
    _添加正文(
        文档,
        "由于509只历史成分股中仅289只有行情覆盖，并进一步排除9只OHLC异常股票，"
        "实际研究仅使用280只股票，存在样本选择偏差。此外，前复权点时性和真实涨跌停"
        "成交状态无法完全验证。因此，本文结果用于展示严谨的量化研究流程，不应直接视为"
        "可执行投资策略业绩。",
    )
    关键词 = 文档.add_paragraph(
        "关键词：Alpha101；沪深300；横截面收益预测；LightGBM；SHAP；机器学习"
    )
    关键词.runs[0].bold = True

    文档.add_heading("1 研究背景与问题", level=1)
    _添加正文(
        文档,
        "量化选股的核心任务是在同一交易时点对股票未来相对表现进行排序。传统线性因子"
        "模型具有透明、稳定和便于解释的优点，但难以捕捉因子之间的非线性关系。Alpha101"
        "提供了一组基于价格与成交量的公式化信号，而树模型能够进一步学习因子阈值与交互。"
        "本文关注两个问题：精选Alpha101因子是否具有样本外预测能力；非线性机器学习是否"
        "显著优于线性组合。",
    )
    _添加正文(
        文档,
        "与单纯追求分类准确率不同，本文同时使用Rank IC、五分组收益、Top-Bottom收益、"
        "换手率、交易成本和年度稳定性评价。模型结果无论正负均完整保留，测试集不参与"
        "因子方向选择和超参数反复调整。",
    )

    文档.add_heading("2 数据、股票池与清洗", level=1)
    _添加正文(
        文档,
        "原始Choice行情覆盖2019年1月2日至2025年12月31日，共1,699个交易日。"
        "每只股票包含开盘价、收盘价、最高价、最低价、成交量和成交金额。OHLC采用前复权"
        "口径，成交量和成交金额保留原始值。利用2019年1月2日的300只成分快照及随后498条"
        "成分进出记录，重建了16个调整日的动态股票池，历史成分并集为509只。",
    )
    _添加正文(
        文档,
        "行情文件实际固定覆盖300只较新成分，其中只有289只属于2019—2025年历史成分并集。"
        "清洗进一步发现9只股票长期存在最低价高于开盘或收盘等OHLC逻辑异常，最终可用代码"
        "为280只。零价格转换为缺失，停牌和上市前日期保留但不填充，股票—日期主键无重复。"
        "每日实际动态样本约181—281只。",
    )
    _添加正文(
        文档,
        "这一数据限制是全文最重要的外部有效性约束。当前样本更接近“可得沪深300历史成分"
        "子样本”，不能替代完整的点时沪深300数据库。为避免制造虚假精度，本文没有模拟或"
        "插补缺失220只股票的历史行情。",
    )

    文档.add_heading("3 因子构造与预处理", level=1)
    _添加正文(
        文档,
        "本文复现Alpha2、3、4、6、8、9、10、12、13、14、15、16、18、20、22、23、"
        "33和34。底层运算符包括delay、delta、滚动求和、均值、极值、标准差、时间序列排名、"
        "相关系数、协方差及横截面排名。时间序列运算在股票内部按统一市场交易日进行，窗口"
        "不足时返回缺失；横截面排名只使用当日动态可得成分子样本。",
    )
    _添加正文(
        文档,
        "原始因子在每个交易日分别按1%和99%分位缩尾，再进行横截面Z标准化。滚动窗口产生的"
        "初始缺失不填零。Alpha15因短窗口相关系数在排名不变时无法定义，缺失率约14.6%，"
        "模型阶段使用仅在训练集拟合的中位数填补。未来扰动测试确认，修改t+1以后行情不会"
        "改变t日及以前因子。",
    )

    文档.add_heading("4 标签、特征与样本划分", level=1)
    _添加正文(
        文档,
        "主连续标签为Open(t+6)/Open(t+1)-1，再减去同一信号日可得动态子样本收益中位数。"
        "相对收益大于0记为1，否则记为0。该定义使分类比例在训练、验证、测试期均接近50%，"
        "避免模型仅学习市场整体涨跌方向。另保留1日和10日相对收益用于稳健性检验。",
    )
    _添加正文(
        文档,
        "除18个Alpha外，构造1/5/20日收益率、5/20日波动率、日内振幅、成交量相对20日"
        "均量、成交金额对数、成交金额相对20日均值和开盘至收盘收益，共28个特征。训练集、"
        "验证集、测试集样本分别为174,080、51,353和118,270。切分同时约束信号日、建仓日"
        "和退出日，边界样本被purge，避免5日标签跨越数据集边界。",
    )

    文档.add_heading("5 单因子有效性", level=1)
    _添加正文(
        文档,
        "每日Rank IC定义为因子值与未来5日相对收益的Spearman相关系数。汇总指标包括IC均值、"
        "IC标准差、ICIR、正IC比例和Newey-West/HAC t值。由于5日标签每日重叠，HAC最大滞后"
        "设为5，避免将相邻日期误认为独立样本。",
    )
    _添加数据表(
        文档,
        因子结果.head(10),
        ["因子", "IC均值", "ICIR", "正IC比例", "HAC_t值", "Q5-Q1"],
    )
    _添加图片(
        文档,
        图表目录 / "factor_ic.png",
        "图1 测试集主要Alpha101因子Rank IC均值",
    )
    _添加正文(
        文档,
        "Alpha8的测试集IC均值为0.0338、Q5-Q1为0.297%；Alpha6的IC均值为0.0285；"
        "Alpha16虽然IC均值为0.0238，但ICIR达到0.2375且正IC比例为58.5%。部分因子的"
        "IC与分组收益方向并不完全一致，说明短期收益分布、极端值和因子非线性会影响经济表现。",
    )

    文档.add_heading("6 模型设计与样本外结果", level=1)
    _添加正文(
        文档,
        "Logistic Regression使用训练期中位数填补和标准化；Random Forest限制最大深度并"
        "设置较大的叶节点最小样本数；LightGBM使用较低学习率、15或31个叶节点和正则化。"
        "超参数仅根据2023年验证集ROC-AUC选择，模型不在查看测试集后重新拟合。",
    )
    _添加数据表(
        文档,
        模型结果,
        ["模型", "ROC_AUC", "PR_AUC", "每日等权AUC", "预测RankIC均值"],
    )
    _添加图片(
        文档,
        图表目录 / "model_auc.png",
        "图2 2024—2025年测试集模型ROC-AUC",
    )
    _添加正文(
        文档,
        "完整Random Forest取得最高ROC-AUC 0.5158，完整LightGBM为0.5153，差异很小。"
        "Alpha Logistic达到0.5138并优于Alpha LightGBM，说明非线性模型并非自动优于"
        "线性模型。普通量价Logistic的AUC仅0.5013，加入Alpha因子后指标上升，支持Alpha"
        "因子提供增量信息这一结论。金融市场横截面信号噪声较大，0.51左右的AUC应结合"
        "Rank IC和组合排序评价，而不能按一般分类任务标准判断。",
    )

    文档.add_heading("7 SHAP解释与特征重要性", level=1)
    _添加正文(
        文档,
        "以完整LightGBM作为主要解释对象，在2024和2025年分别抽样3,000条记录计算TreeSHAP，"
        "并以测试集置换重要性进行交叉验证。SHAP反映模型预测分配，不代表因果效应；相关因子"
        "之间可能共享或重新分配重要性。",
    )
    _添加数据表(
        文档,
        SHAP结果.head(10),
        ["特征", "平均绝对SHAP"],
        小数位数=5,
    )
    _添加图片(
        文档,
        项目根目录 / "outputs/shap/shap_importance_bar.png",
        "图3 完整LightGBM全局SHAP重要性",
    )
    _添加图片(
        文档,
        项目根目录 / "outputs/shap/shap_summary.png",
        "图4 完整LightGBM SHAP分布",
    )
    _添加正文(
        文档,
        "20日收益率和20日波动率位居前两位，说明中短期趋势和风险状态是重要预测条件。Alpha6、"
        "Alpha13、Alpha33、Alpha18、Alpha9、Alpha8和Alpha12进入前十，表明模型同时使用"
        "价量关系、横截面协方差和短期反转信息。成交金额对数排名较高，提示流动性暴露仍需"
        "谨慎解释。",
    )

    文档.add_heading("8 分组收益与交易成本", level=1)
    _添加数据表(
        文档,
        分组结果.sort_values("Top-Bottom", ascending=False),
        ["模型", "Top20收益", "Bottom20收益", "Top-Bottom", "Top20相对收益"],
    )
    _添加正文(
        文档,
        "测试集每日按照预测概率分为Top和Bottom 20%。Alpha Logistic的Top-Bottom五日平均"
        "收益差最高，为0.283%；完整Random Forest为0.266%；完整LightGBM为0.236%。普通"
        "量价Logistic的收益差为负，再次显示量价特征单独使用时缺乏稳定排序能力。",
    )
    _添加数据表(
        文档,
        成本结果,
        ["双边成本基点", "累计收益", "年化收益", "夏普比率", "最大回撤", "胜率"],
    )
    _添加图片(
        文档,
        项目根目录 / "outputs/backtest/weekly_relative_nav.png",
        "图5 完整LightGBM Top20%非重叠组合相对净值",
    )
    _添加正文(
        文档,
        "为减少5日收益重叠，每5个交易日选取一次完整LightGBM Top20%等权组合，共96期。"
        "在0、10、20和30bps双边成本情景下，累计相对收益分别为57.5%、47.9%、38.8%和"
        "30.2%。这里的“相对”基准是可得子样本中位数，不是官方沪深300指数；换手成本采用"
        "简化基点假设，未模拟涨跌停无法成交、冲击成本和最低佣金，因此夏普比率不可与真实"
        "产品业绩直接比较。",
    )

    文档.add_heading("9 稳健性检验", level=1)
    _添加数据表(
        文档,
        周期结果,
        ["预测周期", "RankIC均值", "正IC比例", "有效交易日"],
    )
    _添加正文(
        文档,
        "同一完整LightGBM分数对未来1日、5日和10日相对收益的Rank IC均值分别为0.0164、"
        "0.0295和0.0370，方向一致且10日更高。模型在2024和2025年均单独评价，结果文件"
        "保留年度AUC和年度分组收益。特征消融显示Alpha Logistic明显优于普通量价Logistic，"
        "完整模型相较Alpha单独模型的提升有限，说明额外量价特征并非唯一性能来源。",
    )
    _添加正文(
        文档,
        "成本从0提高至30bps后相对收益和夏普连续下降，但结果方向未反转。需要强调，多个"
        "稳健性方案之间并非完全独立，本文未对全部比较实施严格多重检验校正，因此应将其作为"
        "稳定性证据而非新的显著性发现。",
    )

    文档.add_heading("10 局限性与风险", level=1)
    for 文本 in [
        "第一，历史成分股行情覆盖率仅56.78%，已退出指数的股票大量缺失，无法完全消除幸存者偏差。",
        "第二，Choice前复权序列可能根据后续公司行为回溯调整，缺少原始复权因子使点时性无法彻底验证。",
        "第三，缺少历史涨跌停价、停复牌状态细分、盘口价差和融券可得性，交易成本只能做情景分析。",
        "第四，动态子样本每日数量变化，横截面中位数不是官方沪深300指数收益，组合结果不能直接称为指数增强业绩。",
        "第五，模型AUC约0.51，信号较弱；在不同数据供应商、复权方法和完整股票池上可能发生明显变化。",
        "第六，SHAP是预测解释而非因果识别，特征重要性不等于可独立交易的因子收益。",
    ]:
        _添加正文(文档, 文本)

    文档.add_heading("11 结论", level=1)
    _添加正文(
        文档,
        "在当前可得历史成分子样本中，精选Alpha101因子表现出弱但可检测的样本外横截面预测"
        "能力。Alpha8、Alpha6、Alpha18和Alpha16具有较稳定的单因子IC。Random Forest和"
        "完整LightGBM取得最高分类指标，但Alpha Logistic仍然具有竞争力，说明模型复杂度并非"
        "决定性能的唯一因素。",
    )
    _添加正文(
        文档,
        "量价特征单独构成的线性基准接近随机，而加入Alpha因子后AUC、Rank IC和Top-Bottom"
        "收益均改善。SHAP进一步表明，模型同时依赖中期动量与波动、价量关系和流动性特征。"
        "总体而言，项目支持“精选Alpha101因子具有一定增量信息，但预测能力有限且对数据口径"
        "敏感”的结论。",
    )
    _添加正文(
        文档,
        "未来最重要的改进不是继续增加模型复杂度，而是补齐509只历史成分股的点时行情、原始"
        "价格与复权因子，并加入真实交易状态和官方沪深300全收益指数。完成这些数据改进后，"
        "再采用滚动训练和容量约束进行可执行指数增强研究。",
    )

    文档.add_heading("参考文献", level=1)
    参考文献 = [
        "[1] Kakushadze, Z. 101 Formulaic Alphas. Wilmott, 2016.",
        "[2] Breiman, L. Random Forests. Machine Learning, 2001.",
        "[3] Ke, G. et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS, 2017.",
        "[4] Lundberg, S. M., Lee, S. I. A Unified Approach to Interpreting Model Predictions. NeurIPS, 2017.",
        "[5] Newey, W. K., West, K. D. A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. Econometrica, 1987.",
    ]
    for 条目 in 参考文献:
        段落 = 文档.add_paragraph(条目)
        段落.paragraph_format.line_spacing = 1.3

    文档.add_section(WD_SECTION.NEW_PAGE)
    报告目录.mkdir(parents=True, exist_ok=True)
    文档.save(报告文件)
    print(f"正式研究报告已生成: {报告文件}")


if __name__ == "__main__":
    主程序()
