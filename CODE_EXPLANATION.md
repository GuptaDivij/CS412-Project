# preprocessing.py 代码解释

## 📋 整体设计思路

这个预处理脚本采用**面向对象的设计**，将整个数据预处理流程封装在一个类中。主要思路是：

1. **数据加载** → 加载所有相关数据集
2. **数据清洗** → 去除无效数据，处理缺失值
3. **特征工程** → 从原始数据创建有意义的特征
4. **数据准备** → 准备用于机器学习模型的格式
5. **降维分析** → 分析是否需要降维（PCA）
6. **可视化** → 帮助理解特征的重要性

---

## 🏗️ 类结构

### `MarchMadnessPreprocessor` 类

**核心属性：**
- `self.data`: 存储所有加载的数据集（字典）
- `self.features`: 处理后的特征矩阵
- `self.target`: 目标变量（胜负结果）

---

## 📦 函数详解

### 1. `__init__(self, data_dir='march-machine-learning-mania-2025')`
**作用：** 初始化预处理器

**功能：**
- 设置数据目录路径
- 初始化数据存储字典
- 初始化特征和目标变量

**为什么重要：** 创建类的实例，为后续所有操作做准备

---

### 2. `load_data(self)`
**作用：** 加载所有相关数据集

**加载的数据：**
- **核心数据**（必需）：
  - `MTeams.csv` - 球队信息
  - `MRegularSeasonCompactResults.csv` - 常规赛季比赛结果
  - `MNCAATourneyCompactResults.csv` - 锦标赛历史结果
  - `MNCAATourneySeeds.csv` - 锦标赛种子排名
  - `MSeasons.csv` - 赛季信息

- **可选数据**（增强特征）：
  - `MRegularSeasonDetailedResults.csv` - 详细比赛统计
  - `MMasseyOrdinals.csv` - 排名系统数据
  - `MTeamConferences.csv` - 球队联盟信息
  - `MConferenceTourneyGames.csv` - 联盟锦标赛

**设计思路：**
- 核心数据必须加载成功，否则返回 False
- 可选数据如果缺失不影响主流程，只给出警告
- 使用 try-except 优雅处理缺失文件

---

### 3. `analyze_missing_values(self)`
**作用：** 分析所有数据集中的缺失值

**功能：**
- 统计每个数据集的缺失值数量和百分比
- 只显示有缺失值的列
- 帮助了解数据质量

**输出示例：**
```
MREGULAR_SEASON:
  Total rows: 192,912
  Missing Count: 0 ✓

MTOURNEY_SEEDS:
  Total rows: 2,608
  Missing Count: 0 ✓
```

**为什么重要：** 在数据清洗前了解缺失值情况，决定处理策略

---

### 4. `clean_data(self)`
**作用：** 清洗数据，去除噪声和无效数据

**清洗规则：**

1. **常规赛季数据：**
   - 移除分数 ≤ 0 的比赛
   - 移除无效的球队 ID（不在球队列表中的）

2. **锦标赛数据：**
   - 同样的验证规则

3. **Massey 排名数据：**
   - 只保留每个赛季/系统/球队的最终排名
   - 大幅减少数据量（从 500万+ 行减少到几万行）

4. **详细统计数据：**
   - 移除所有统计项都为 0 或异常的比赛

**设计思路：**
- 数据质量 > 数据数量
- 保留最准确、最新的数据
- 记录清理了多少数据，便于审计

---

### 5. `create_season_stats(self, season=None)`
**作用：** 为每个球队创建赛季级别的聚合统计

**核心逻辑：**
对每个球队的每个赛季，计算：

**基础统计：**
- `Games`: 总比赛数
- `Wins`: 胜场数
- `Losses`: 负场数
- `WinPct`: 胜率 = Wins / Games

**得分统计：**
- `PointsFor`: 总得分
- `PointsAgainst`: 总失分
- `AvgPointsFor`: 平均得分
- `AvgPointsAgainst`: 平均失分

**分差统计：**
- `AvgScoreDiff`: 平均分差 = (总净胜分) / 比赛数
  - 计算方式：赢球时的分差 - 输球时的分差

**场地统计：**
- `HomeWins`: 主场胜场
- `AwayWins`: 客场胜场
- `NeutralWins`: 中立场地胜场

**设计思路：**
- 从原始比赛结果（单场比赛）→ 聚合统计（球队级别）
- 这些统计是后续特征工程的基础
- 可以指定特定赛季，也可以处理所有赛季

**关键计算说明：**
```python
# 赢球时：分差 = WScore - LScore (正数)
# 输球时：分差 = WScore - LScore (但 WScore 是对手的，LScore 是自己的，所以是负数)
# 总净胜分 = 所有赢球分差的和 - 所有输球分差的和
```

---

### 6. `create_matchup_features(self, season_stats, tourney_seeds=None)`
**作用：** 创建用于预测的特征矩阵（每行代表一个 matchup）

**核心思路：**
从锦标赛历史结果中，为每场比赛创建两个方向的特征：
- Team1 vs Team2（Team1 获胜）
- Team2 vs Team1（Team2 失败）

**特征类型：**

1. **个体特征：**
   - `Team1_WinPct`, `Team2_WinPct`
   - `Team1_AvgPointsFor`, `Team2_AvgPointsFor`
   - `Team1_AvgPointsAgainst`, `Team2_AvgPointsAgainst`
   - `Team1_AvgScoreDiff`, `Team2_AvgScoreDiff`
   - `Team1_Games`, `Team2_Games`
   - `Team1_Seed`, `Team2_Seed`（如果可用）

2. **差异特征（最重要）：**
   - `WinPct_Diff` = Team1_WinPct - Team2_WinPct
   - `PointsFor_Diff` = Team1_AvgPointsFor - Team2_AvgPointsFor
   - `PointsAgainst_Diff` = Team1_AvgPointsAgainst - Team2_AvgPointsAgainst
   - `ScoreDiff_Diff` = Team1_AvgScoreDiff - Team2_AvgScoreDiff
   - `Seed_Diff` = Team1_Seed - Team2_Seed

3. **目标变量：**
   - `Outcome`: 1 = Team1 获胜，0 = Team1 失败

**为什么创建双向特征：**
- 增强训练数据量
- 让模型学习对称性（A vs B 和 B vs A 是对称的）
- 提高模型泛化能力

**种子处理：**
- 原始格式："W01", "X16" 等
- 提取数字部分：1, 16 等
- 种子越低，球队越强

---

### 7. `handle_missing_values(self, df)`
**作用：** 处理特征矩阵中的缺失值

**处理策略：**
- **种子数据：** 用中位数填充（因为种子是 1-16 的整数）
- **其他数值特征：** 用中位数填充（中位数对异常值不敏感）

**设计思路：**
- 不删除含缺失值的行（会损失数据）
- 使用中位数而非均值（更稳健）
- 确保所有特征都有值，模型可以正常训练

---

### 8. `prepare_features_for_modeling(self, features_df)`
**作用：** 准备最终的模型输入格式

**功能：**
1. 分离特征（X）和目标（y）
2. 移除非特征列（Season, Team1, Team2, Outcome）
3. 处理缺失值
4. 存储到 `self.features` 和 `self.target`

**返回：**
- `X`: 特征矩阵（DataFrame）
- `y`: 目标向量（Series）

**为什么重要：** 这是模型训练的最终输入格式

---

### 9. `analyze_dimensionality(self, X, threshold=0.95)`
**作用：** 分析特征维度，判断是否需要降维

**分析内容：**
1. **特征数量：** 有多少个特征
2. **样本数量：** 有多少个训练样本
3. **特征-样本比：** 比例是否合理
4. **高相关性特征对：** 找出相关性 > 0.9 的特征（冗余）

**建议：**
- 如果特征数 > 10，建议考虑 PCA
- 如果发现高相关性特征，建议删除冗余特征

**设计思路：** 预防维度灾难，提高模型效率

---

### 10. `apply_pca(self, X, n_components=None, variance_threshold=0.95)`
**作用：** 应用主成分分析（PCA）进行降维

**流程：**
1. **标准化：** 使用 StandardScaler（PCA 对尺度敏感）
2. **确定组件数：**
   - 如果指定 `n_components`，使用指定值
   - 否则，找到能解释 `variance_threshold`（默认 95%）方差的最小组件数
3. **应用 PCA：** 将原始特征转换为主成分
4. **返回：** 转换后的特征矩阵和 PCA 模型

**设计思路：**
- 减少特征数量，降低计算复杂度
- 保留大部分信息（95% 方差）
- 可用于可视化或加速训练

---

### 11. `get_summary(self)`
**作用：** 打印所有数据的摘要信息

**显示内容：**
- 每个数据集的形状和列名
- 赛季范围
- 特征矩阵和目标变量的信息

**为什么重要：** 快速了解数据概况，便于调试和验证

---

## 🎨 可视化函数

### 12. `plot_feature_correlation(self, X, ...)`
**作用：** 绘制特征相关性热力图

**功能：**
- 显示特征之间的相关性矩阵
- 识别高度相关的特征对（|r| > 0.7）
- 帮助发现冗余特征

**用途：** 特征选择时，删除高度相关的特征之一

---

### 13. `plot_feature_importance(self, X, y, method='random_forest', ...)`
**作用：** 分析并可视化特征重要性

**三种方法：**
1. **correlation：** 特征与目标的相关性（最快）
2. **random_forest：** 随机森林的特征重要性（最准确）
3. **mutual_info：** 互信息（捕捉非线性关系）

**用途：** 识别最重要的特征，用于特征选择

---

### 14. `plot_feature_distributions(self, X, y, ...)`
**作用：** 对比获胜和失败时的特征分布

**功能：**
- 显示每个特征在 Team1 获胜 vs Team2 获胜时的分布
- 如果分布差异明显，说明特征有用

**用途：** 直观理解哪些特征能区分胜负

---

### 15. `plot_pca_variance(self, X, ...)`
**作用：** 分析 PCA 的方差解释情况

**显示：**
- 每个主成分解释的方差
- 累积方差解释曲线
- 达到 90% 和 95% 方差需要的组件数

**用途：** 决定是否需要 PCA，以及需要多少个主成分

---

### 16. `plot_feature_vs_target(self, X, y, ...)`
**作用：** 用箱线图显示特征与目标的关系

**功能：**
- 显示每个特征在不同结果下的分布
- 箱线图显示中位数、四分位数、异常值

**用途：** 理解特征如何影响预测结果

---

### 17. `visualize_all(self, X, y, ...)`
**作用：** 一键运行所有可视化

**功能：** 按顺序执行所有可视化函数，全面了解特征

**用途：** 快速探索特征，无需逐个调用

---

## 🔄 完整工作流程

```
1. 初始化 → __init__()
2. 加载数据 → load_data()
3. 分析缺失值 → analyze_missing_values()
4. 清洗数据 → clean_data()
5. 创建赛季统计 → create_season_stats()
6. 创建 matchup 特征 → create_matchup_features()
7. 准备模型输入 → prepare_features_for_modeling()
8. 分析维度 → analyze_dimensionality()
9. （可选）应用 PCA → apply_pca()
10. 可视化特征 → visualize_all()
11. 查看摘要 → get_summary()
```

---

## 🎯 设计亮点

### 1. **模块化设计**
- 每个函数职责单一，易于测试和维护
- 可以单独调用任何函数

### 2. **容错处理**
- 可选数据缺失不影响主流程
- 库缺失时给出友好提示

### 3. **特征工程思路**
- 从原始数据 → 聚合统计 → 差异特征
- 差异特征比绝对特征更有预测力（如 WinPct_Diff）

### 4. **数据增强**
- 为每场比赛创建双向特征（A vs B 和 B vs A）
- 增加训练数据量，提高模型性能

### 5. **可扩展性**
- 易于添加新的特征类型
- 可视化函数可以独立使用

---

## 💡 使用建议

### 基本流程：
```python
preprocessor = MarchMadnessPreprocessor()
preprocessor.load_data()
preprocessor.clean_data()
season_stats = preprocessor.create_season_stats()
features_df = preprocessor.create_matchup_features(season_stats, ...)
X, y = preprocessor.prepare_features_for_modeling(features_df)
```

### 特征选择流程：
```python
# 1. 可视化了解特征
preprocessor.visualize_all(X, y)

# 2. 根据重要性选择特征
importance = preprocessor.plot_feature_importance(X, y, method='random_forest')
top_features = importance.head(10).index.tolist()
X_selected = X[top_features]

# 3. 如果特征太多，考虑 PCA
if len(X.columns) > 20:
    X_pca, _ = preprocessor.apply_pca(X, variance_threshold=0.95)
```

---

## 🔍 关键设计决策

1. **为什么使用差异特征？**
   - 相对比较比绝对数值更有预测力
   - 例如：WinPct_Diff 比单独的 WinPct 更有用

2. **为什么创建双向特征？**
   - 增加训练数据
   - 让模型学习对称性

3. **为什么先聚合再创建特征？**
   - 减少噪声
   - 赛季级别的统计更稳定
   - 单场比赛的波动太大

4. **为什么使用中位数填充缺失值？**
   - 对异常值不敏感
   - 比均值更稳健

---

这个预处理脚本设计得既全面又灵活，可以处理从数据加载到特征准备的全流程，并且提供了丰富的可视化工具帮助理解数据。





