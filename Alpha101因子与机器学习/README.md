# 基于精选Alpha101因子与机器学习的沪深300股票收益预测

英文题目：*Cross-Sectional Return Prediction for CSI 300 Stocks Using Selected Alpha101 Factors and Machine Learning*

## 项目目标

本项目研究仅依赖OHLCV数据的精选Alpha101因子能否预测沪深300股票未来5个完整交易日的横截面相对收益，并比较Logistic Regression、Random Forest和LightGBM的线性及非线性因子组合能力。

信号在交易日`t`收盘后生成，计划于`t+1`开盘建仓、`t+6`开盘退出。连续标签为个股5日收益减去同日可得动态成分子样本收益中位数，分类标签为相对收益是否大于0。

## 数据范围与重要限制

- 原始区间：2019-01-02至2025-12-31，共1,699个交易日。
- 历史沪深300成分股并集：509只。
- 现有行情仅覆盖其中289只，覆盖率56.78%。
- 排除9只长期OHLC逻辑异常股票后，实际使用280只股票。
- 每日可得动态成分子样本约181—281只。
- 结果存在明显样本选择偏差，只能视为探索性研究，不能解释为完整无偏的沪深300回测。
- 前复权价格可能包含供应商回溯调整，本项目无法完全验证点时复权因子。
- 缺少逐笔成交、历史涨跌停状态及融券可得性，交易成本结果属于情景分析。

## 已实现内容

1. Choice多层宽表转股票—日期长表；
2. 历史动态成分股链条重建；
3. 零值、停牌、重复记录及OHLC逻辑检查；
4. Alpha101基础时间序列和横截面运算符；
5. 18个精选Alpha101因子；
6. 因子按日1%/99%缩尾和横截面Z标准化；
7. 10个普通量价特征；
8. 未来1日、5日、10日收益及相对收益；
9. 2019—2022训练、2023验证、2024—2025测试及边界purge；
10. 单因子Rank IC、ICIR、HAC t值和五分组收益；
11. Logistic、Random Forest、LightGBM及特征消融比较；
12. SHAP、置换重要性、分年检验和交易成本敏感性；
13. 每5日非重叠Top20%组合分析。

## 18个精选因子

`Alpha2、Alpha3、Alpha4、Alpha6、Alpha8、Alpha9、Alpha10、Alpha12、Alpha13、Alpha14、Alpha15、Alpha16、Alpha18、Alpha20、Alpha22、Alpha23、Alpha33、Alpha34`

这些因子不依赖VWAP、行业中性化或市值字段。因子公式以Zura Kakushadze的《101 Formulaic Alphas》为准。

## 样本外主要结果

- 完整Random Forest测试集ROC-AUC为0.5158，预测Rank IC均值为0.0308；
- 完整LightGBM测试集ROC-AUC为0.5153，预测Rank IC均值为0.0295；
- Alpha Logistic测试集ROC-AUC为0.5138；
- 普通量价Logistic测试集ROC-AUC仅0.5013；
- 表现较好的单因子包括Alpha8、Alpha6、Alpha18和Alpha16；
- 完整LightGBM中重要特征包括20日收益率、20日波动率、Alpha6、成交金额对数、Alpha13和Alpha33；
- 完整LightGBM测试集Top-Bottom五日平均收益差约0.236%；
- 20bps成本情景下，每5日非重叠Top20%组合累计相对收益约38.8%。

上述收益以可得动态子样本中位数为相对基准，不是相对于官方沪深300全收益指数的真实超额收益。

## 目录说明

- `src/csi300_ml/`：数据、因子、标签、评价和模型模块；
- `scripts/`：按执行顺序编号的研究脚本；
- `data/processed/`：Parquet中间数据；
- `models/`：冻结模型；
- `outputs/`：Excel结果、Parquet明细和图表；
- `reports/`：正式研究报告；
- `tests/`：基础运算符和因子未来扰动测试。

## 运行方法

安装依赖：

```bash
python -m pip install -r requirements.txt
```

按顺序运行：

```bash
python scripts/01_data_audit.py
python scripts/02_build_clean_panel.py
python scripts/03_build_first_alpha_batch.py
python scripts/05_build_all_alpha_factors.py
python scripts/06_build_model_dataset.py
python scripts/07_evaluate_factors.py
python scripts/08_train_models.py
python scripts/09_explain_model.py
python scripts/10_backtest_robustness.py
```

`scripts/04_export_factor_excel.py`仅用于把第一批因子Parquet转换为Excel查看版。

## 核心结果文件

- `outputs/factor_evaluation/factor_evaluation.xlsx`
- `outputs/models/model_metrics.xlsx`
- `outputs/shap/feature_importance.xlsx`
- `outputs/backtest/backtest_robustness.xlsx`
- `outputs/backtest/weekly_relative_nav.png`
- `outputs/shap/shap_summary.png`

## 结论

在当前不完整历史成分子样本中，精选Alpha101因子具有弱但可检测的横截面预测能力。非线性模型并未全面压倒线性模型：Random Forest和完整LightGBM取得最高分类指标，但Alpha Logistic同样具有竞争力。普通量价特征单独使用时接近随机，加入Alpha因子后预测能力改善，说明性能并非仅来自模型复杂度。

金融市场信号本身较弱，0.51左右的AUC并不异常。当前结果的主要限制不是模型，而是历史成分行情覆盖不足、前复权点时性无法完全验证，以及交易可执行性字段缺失。
