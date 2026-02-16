# ONDS 多信号量化研究框架

针对 **Ondas Holdings (ONDS)** 的综合量化研究，涵盖 12 种信号来源、30 种交易策略、参数优化、以及 Alpha 衰减检测。结合 NLP 情绪分析（Reddit、新闻）、技术指标、跨资产信号（黄金、白银、比特币、VIX）、暗池数据、期权/IV 分析、同业板块分析（无人机/国防同业）、事件研究与市场状态检测。

## 核心发现

### 进阶策略表现（30 种策略，2025-01 至 2026-02）

| 策略 | Sharpe | 报酬率 | 最大回撤 | 胜率 | 排列检验 p 值 |
|------|--------|--------|---------|------|-------------|
| **ML 方向预测 (top-15)** | **+4.06** | +314% | -26% | 61.8% | **0.009** |
| **动量+停损 (波动率调整)** | **+2.39** | +61% | -11% | 54.4% | **0.021** |
| **自适应集成** | **+1.79** | +71% | -22% | 54.6% | **0.039** |
| 市场逆势 | +1.24 | +88% | -35% | 58.5% | - |
| 市场逆势 (波动率调整) | +1.11 | +24% | -8% | 58.5% | 0.064 |
| 组合最佳 | +1.09 | +41% | -18% | 52.9% | - |
| 多时间框架动量 (波动率调整) | +1.02 | +34% | -16% | 50.9% | - |

**6 种策略 Sharpe > 1.0**，其中 3 种在 5% 显著性水平通过排列检验（2000 次随机打乱）。

### 稳健性验证

| 策略 | Bootstrap Sharpe | 95% 信赖区间 | P(Sharpe>0) |
|------|-----------------|-------------|-------------|
| ML 方向预测 | +2.20 | [+0.39, +3.85] | 98.9% |
| 动量+停损 (波动率调整) | +1.93 | [-0.21, +3.86] | 96.3% |
| 自适应集成 | +1.49 | [-0.33, +3.28] | 94.5% |
| 市场逆势 (波动率调整) | +1.04 | [-0.78, +2.85] | 87.3% |

### 参数优化结果（60/10/30 分割）

将 280 天资料切为训练（60%）/ 验证（10%）/ 测试（30%）三段，进行网格搜索优化：

| 策略 | 最佳参数 | 测试 Sharpe | 测试报酬 | Alpha | 训练 Sharpe | 验证 Sharpe |
|------|---------|------------|---------|-------|------------|------------|
| ML 方向预测 | depth=5, feat=20, thr=0.58 | +2.77 | +48.9% | +35.4% | +0.85 | 正 |
| 组合最佳 | peer=0.2, contra=0.4, mtf=0.3, mr=0.3 | +2.07 | +22.8% | +9.3% | +1.66 | 正 |
| 均值回归 | z=1.5, lookback=10 | +0.95 | +17.5% | +4.0% | +0.18 | 正 |
| 动量+停损 | SMA 15/50, trail=15%, loss=10% | +0.39 | +8.8% | -4.7% | +7.20 | 正 |

**关键发现**：Peer Lead-Lag 策略在验证期完全失效，被 60/10/30 过滤器淘汰——证实该分割有效防止过拟合。

### Alpha 衰减健康仪表板

| 策略 | 健康评等 | 近期 Sharpe | 近期 IC | 胜率 | 获利因子 | Alpha 斜率 | OU 半衰期 |
|------|---------|------------|---------|------|---------|-----------|----------|
| Peer Lead-Lag | HEALTHY | +2.21 | +0.22 | 51.7% | 2.14 | -0.0003 | 107 天 |
| 组合最佳 | HEALTHY | +0.06 | +0.20 | 58.3% | 1.71 | -0.0009 | 168 天 |
| 均值回归 | MONITOR | +2.45 | +0.13 | 35.0% | 1.45 | -0.0006 | 121 天 |
| 动量+停损 | MONITOR | +1.48 | 0.00 | 45.0% | 1.31 | -0.0007 | 103 天 |
| ML 方向预测 | RECALIBRATE | +1.93 | -0.14 | 38.3% | 2.59 | -0.0013 | 119 天 |

- **HEALTHY（健康）**：4+ 个指标亮绿灯，可继续使用
- **MONITOR（观察）**：需密切监控，部分指标转弱
- **RECALIBRATE（需重新校准）**：3+ 个指标亮红灯，建议重新调参或停用

## 研究方法论

### 一、资料收集（Plan 0）

```
资料来源
├── yfinance: ONDS + 16 档相关标的（280 个交易日，2025-01 起）
├── Reddit JSON API: 来自 7 个子版的 362 篇贴文
├── Google News RSS: 100+ 篇新闻文章
├── Finnhub API: 217 篇新闻 + 分析师评等
├── ChartExchange: ONDS FINRA 空头成交量（375 天）
├── yfinance Options: 529 份合约、10 个到期日
└── 暗池资料: SqueezeMetrics DIX/GEX（3,720 天）
```

### 二、分析模组（Plans 1-11）

| Plan | 分析主题 | 说明 |
|------|---------|------|
| 1 | 技术分析 | RSI、MACD、布林通道、ATR、OBV、SMA 交叉 |
| 2 | 同业板块 | RCAT、AVAV、KTOS、JOBY、LMT、RTX 领先/落后关系 |
| 3 | 跨资产 | 黄金、白银、BTC、VIX、SPY、QQQ、TLT 相关性 |
| 4 | 暗池 | DIX/GEX 信号 + ChartExchange FINRA 空头成交量 |
| 5 | Reddit 情绪 | VADER + FinBERT 集成情绪评分 |
| 6 | 新闻情绪 | Google News + NLP 情绪分析 |
| 7 | CEO Twitter | 框架已建（需 X 平台认证） |
| 8 | 分析师报告 | Finnhub 共识评等 1.29（极度看多） |
| 9 | 事件研究 | 12 个 ONDS 事件的累积异常报酬（CAR）分析 |
| 10 | 期权/IV | IV Smile、IV Surface、偏度、期限结构 |
| 11 | 状态检测 | HMM、GMM、Markov Switching、波动率状态 |

### 三、融合模组（Plan 12）

- 103 维特征矩阵（技术 + 跨资产 + 板块 + 情绪 + 状态 + 空头成交量）
- 随机森林 & 梯度提升（时间序列交叉验证）
- 特征重要度排序 + 前向滚动回测

### 四、进阶研究（Plan 13）

- 106 维特征（防止资料泄漏的工程处理）
- 8 个分类实验（方向、缺口、日内、三分类）
- 5 个回归实验（报酬、波动率、区间）
- 15 个基础策略 + 15 个波动率调整版本 = **30 个策略**
- ML 前向滚动信号（RF、GB、XGB、LR）
- Bootstrap 信赖区间（5000 次区块自助法）
- 排列检验（2000 次随机打乱）
- 滚动 Sharpe 稳定性分析

### 五、参数优化（Plan 14）

**目的**：解决「在全样本上调参 = 过拟合」的问题。

**方法 — 60/10/30 分割**：
```
训练期 (60%, 168天)     验证期 (10%, 28天)     测试期 (30%, 84天)
2025-01-02 ~ 09-04      2025-09-05 ~ 10-14      2025-10-15 ~ 2026-02-13
      搜寻参数               早停过滤              最终评估
```

**流程**：
1. 对 7 种策略建立参数化封装（可调 SMA 窗口、z 阈值、ML 超参数等）
2. 对每种策略的参数网格（3~5 个值/参数）进行全组合搜索（共 246 组）
3. 在训练期计算 Sharpe，在验证期计算 Sharpe
4. 挑选**两期都 > 0** 且综合分数最高的参数
5. 用最佳参数在测试期做最终评估

**参数网格范例**：
| 策略 | 可调参数 | 搜索范围 |
|------|---------|---------|
| 动量+停损 | SMA短/长、追踪停损、最大亏损 | 81 组合 |
| 市场逆势 | z 阈值、回顾窗口 | 12 组合 |
| ML 方向 | 特征数、机率阈值、树深度 | 36 组合 |
| 均值回归 | z 阈值、回顾窗口 | 12 组合 |

### 六、Alpha 衰减检测（Plan 15）

**目的**：回答「策略的优势还在吗？何时该停用？」

**四大检测工具**：

1. **滚动资讯系数（Rolling IC）**
   - 策略信号与未来报酬的 Spearman 秩相关系数（40 天滚动窗口）
   - IC 持续正值 = 信号仍有预测力；IC 归零 = 信号失效

2. **累积 Alpha 斜率**
   - 策略超额报酬的累积曲线 + OLS 趋势斜率
   - 斜率为正 = Alpha 仍在累积；斜率为负 = Alpha 正在衰退

3. **OU 半衰期（Ornstein-Uhlenbeck）**
   - 将累积 Alpha 拟合为均值回归过程
   - 半衰期越短 = Alpha 消失越快，需更频繁重新校准

4. **CUSUM 结构性断裂检验**
   - 侦测策略表现的突变点（市场结构改变）
   - 使用布朗桥标准化，5% 显著性水平

**健康仪表板**：每个策略评估 6 个指标，各自标为绿/黄/红灯：
- 近期 Sharpe | 当前回撤 | 胜率 | IC | 获利因子 | Alpha 斜率
- 综合评等：HEALTHY / MONITOR / RECALIBRATE

## 目录结构

```
ONDS_Research/
├── config.py                        # 中央配置（标的、参数）
├── run_all.py                       # 主管线控制器
├── requirements.txt
├── README.md                        # 英文文档
├── README_zh.md                     # 中文文档（本文件）
├── collectors/                      # 资料收集模组
│   ├── prices.py                   # yfinance 股价资料
│   ├── reddit.py                   # Reddit JSON API 爬虫
│   ├── news.py                     # Google News + Finnhub
│   ├── darkpool.py                 # SqueezeMetrics + Stockgrid
│   └── options.py                  # yfinance 期权链
├── analysis/                        # 分析模组
│   ├── technical.py                # 技术指标 + 预测力检验
│   ├── crossasset.py               # 跨资产相关性 + 信号
│   ├── sector.py                   # 同业/板块 领先落后 + 动量
│   ├── darkpool.py                 # DIX/GEX + ChartExchange 空头成交量
│   ├── sentiment.py                # VADER + FinBERT 情绪评分
│   ├── analyst.py                  # 分析师评等分析
│   ├── events.py                   # 事件研究方法论
│   ├── options_iv.py               # IV Smile、Surface、衍生特征
│   ├── regime.py                   # HMM、GMM、Markov Switching
│   ├── fusion.py                   # 多来源特征矩阵 + ML
│   ├── advanced_research.py        # 30 策略、ML 信号、特征选择
│   ├── robustness.py               # Bootstrap、排列检验、稳定性
│   ├── train_test_validation.py    # 70/30 训练测试验证
│   ├── param_optimization.py       # [新] 60/10/30 参数优化
│   └── alpha_decay.py              # [新] Alpha 衰减检测 + 健康仪表板
├── backtests/
│   └── engine.py                   # 共用回测引擎
├── data/
│   ├── raw/                        # 原始资料（不入版控）
│   ├── processed/                  # 处理后资料
│   └── features/                   # ML 特征矩阵
└── output/
    ├── figures/                    # 56+ 张图表
    └── reports/                    # 策略比较、排名、稳健性 CSV
```

## 快速开始

```bash
# 安装套件
pip install -r requirements.txt

# 执行完整管线（下载资料 + 跑全部 12 个分析）
python run_all.py

# 使用已快取的资料（跳过下载）
python run_all.py --skip-collect

# 只跑特定 Plan
python run_all.py --skip-collect --plan 1 3 11 12

# 跑进阶策略研究（30 种策略）
python analysis/advanced_research.py

# 跑稳健性验证（Bootstrap、排列检验）
python analysis/robustness.py

# [新] 跑参数优化（60/10/30 分割）
python analysis/param_optimization.py

# [新] 跑 Alpha 衰减检测
python analysis/alpha_decay.py
```

### 环境变数（选配，用于增强资料来源）
```bash
export FINNHUB_API_KEY="your_key"        # finnhub.io 免费申请
export ALPHA_VANTAGE_KEY="your_key"      # alphavantage.co 免费申请
export REDDIT_CLIENT_ID="your_id"        # PRAW（增强版 Reddit）
export REDDIT_SECRET="your_secret"
```

## 产出文件

### 图表（56+ 张）
- `onds_technical_dashboard.png` — 股价 + RSI + MACD + 成交量
- `sector_comparison.png` — ONDS vs 无人机/国防同业（标准化）
- `crossasset_correlation_heatmap.png` — 完整相关矩阵
- `onds_iv_smile.png` / `onds_iv_surface.png` — IV Smile 与 3D Surface
- `regime_hmm.png` / `regime_volatility.png` — 市场状态叠加图
- `sentiment_reddit_vs_price.png` / `sentiment_news_vs_price.png`
- `feature_importance.png` — 前 N 重要 ML 特征
- `strategy_comparison.png` — 全部策略权益曲线叠加
- `rolling_sharpe_stability.png` — 滚动 60 天 Sharpe
- `param_sensitivity.png` — [新] 参数敏感度热力图
- `optimized_test_equity.png` — [新] 优化后策略在测试期的权益曲线
- `alpha_decay_dashboard.png` — [新] Alpha 衰减仪表板（Sharpe/IC/斜率）
- `cumulative_alpha.png` — [新] 各策略累积 Alpha 曲线
- 30x 回测权益曲线 (`bt_*.png`)

### 报告 CSV
- `strategy_comparison.csv` — 30 种策略并排比较
- `strategy_ranking.csv` — 按 Sharpe 排序
- `advanced_strategy_comparison.csv` — 进阶研究详细比较
- `robustness_report.csv` — Bootstrap CI、排列检验 p 值
- `train_test_validation.csv` — 70/30 训练测试结果
- `param_optimization_results.csv` — [新] 246 组参数搜索结果
- `optimized_test_results.csv` — [新] 优化后测试期表现
- `alpha_decay_report.csv` — [新] 各策略衰减指标与健康评等

## 策略类型详解

### 趋势跟踪类
- **动量+停损**：SMA 短/长交叉确认上升趋势，搭配追踪停损和最大亏损限制
- **多时间框架动量**：5 天/10 天/20 天动量三者一致时才进场

### 均值回归类
- **均值回归**：日报酬率 z 分数超过阈值时反向交易
- **布林通道回归**：股价触及上下轨时反向，搭配成交量确认

### 基本面/另类资料类
- **暗池 DIX 增强**：DIX 滚动百分位信号 + GEX 放大器
- **市场逆势**：NASDAQ/SPY 当日极端表现后，ONDS 次日反向
- **Peer Lead-Lag**：利用 JOBY lag-1（负相关）和 RCAT lag-2 的领先关系

### ML 信号类
- **ML 方向预测**：前向滚动随机森林，使用互资讯筛选的 top-15 特征
- **ML 回归信号**：前向滚动梯度提升回归，按预测幅度调整部位大小

### 集成类
- **组合最佳**：固定权重加权多个子信号
- **自适应集成**：每 20 天根据近 40 天 Sharpe 重新分配权重（指数加权）

### 波动率调整
- 所有策略均可搭配**波动率调整仓位管理**：根据 20 天实现波动率反向调整部位大小，目标年化投组波动率 30%

## 回测方法论

- **执行延迟**：信号在第 t 天产生 → 第 t+1 天执行（Lag-1，避免前视偏差）
- **交易成本**：来回 10 个基点（bps）
- **绩效指标**：Sharpe Ratio、年化报酬/波动率、最大回撤、胜率
- **前向滚动**：ML 使用扩展窗口训练（无未来资料泄漏）
- **稳健性**：区块自助法（5000 次，区块大小=5 天）+ 排列检验（2000 次）

## ONDS 公司概况

**Ondas Holdings (NASDAQ: ONDS)** — 国防/无人机科技公司
- 市值：约 $43.7 亿美元（2026 年初）
- 产品：AURA 无人机系统、FullMAX 无线网路
- 子公司：American Robotics（自主无人机解决方案）
- 关键催化剂：美国国防部合约、北约合作、FAA 无人机法规
- CEO：Eric Brock (@CeoOndas on X)
- 同业：RCAT、AVAV、KTOS、JOBY、LMT、RTX

## 未来改进方向

- **CEO Twitter/X 情绪**：爬取 @CeoOndas 推文并分析情绪（需 X 认证）
- **即时暗池**：串接 SqueezeMetrics 即时 DIX/GEX 资料（需订阅）
- **日内期权流**：监控异常期权活动
- **内部人交易**：追踪 SEC Form 4 申报
- **Reddit 即时串流**：使用 PRAW 即时情绪分析
- **集成调参**：Optuna 超参数优化、SHAP 特征解释、神经网路模型
- **投组优化**：均值方差或风险平价配置

## 授权

学术及研究用途。作为多信号量化分析毕业论文的一部分。
