# 特征可视化指南

这个指南帮助你使用可视化功能来识别哪些特征是有用的。

## 安装依赖

首先确保安装了可视化库：

```bash
pip install matplotlib seaborn
```

或者使用 requirements.txt：

```bash
pip install -r requirements.txt
```

## 快速开始

在 Jupyter notebook 中，运行预处理后，可以使用以下代码：

```python
from preprocessing import MarchMadnessPreprocessor

# ... 运行预处理流程 ...
X, y = preprocessor.prepare_features_for_modeling(features_df)

# 运行所有可视化
preprocessor.visualize_all(X, y)
```

## 可视化功能详解

### 1. 特征相关性热力图 🔥

**功能**：显示特征之间的相关性，帮助识别冗余特征

**使用方法**：
```python
preprocessor.plot_feature_correlation(X, figsize=(14, 12), top_n=15)
```

**解读**：
- 红色区域：正相关（接近1表示高度相关，可能冗余）
- 蓝色区域：负相关
- **建议**：如果两个特征相关性 > 0.7，考虑删除其中一个

### 2. 特征重要性分析 📊

**功能**：识别哪些特征对预测结果最重要

**三种方法**：

#### 方法1：相关性分析（最快）
```python
preprocessor.plot_feature_importance(X, y, method='correlation', top_n=15)
```
- 显示每个特征与目标变量的相关性
- 快速但可能不够准确

#### 方法2：随机森林（最准确）
```python
preprocessor.plot_feature_importance(X, y, method='random_forest', top_n=15)
```
- 使用随机森林模型计算特征重要性
- 更准确但需要更多时间

#### 方法3：互信息（非线性关系）
```python
preprocessor.plot_feature_importance(X, y, method='mutual_info', top_n=15)
```
- 捕捉非线性关系
- 适合复杂的数据关系

**解读**：
- 条越长，特征越重要
- **建议**：重点关注排名前10的特征

### 3. 特征分布对比 📈

**功能**：对比获胜和失败时特征的分布差异

**使用方法**：
```python
preprocessor.plot_feature_distributions(X, y, n_features=9, figsize=(15, 10))
```

**解读**：
- 绿色：Team1 获胜时的特征分布
- 红色：Team2 获胜时的特征分布
- **建议**：如果两个分布差异明显，该特征很有用

### 4. 特征 vs 目标变量 📦

**功能**：使用箱线图显示不同结果下特征的差异

**使用方法**：
```python
preprocessor.plot_feature_vs_target(X, y, top_n=9, figsize=(15, 10))
```

**解读**：
- 箱线图显示中位数、四分位数和异常值
- 如果两个箱线图差异明显，该特征有用
- 标题显示与目标的相关性

### 5. PCA 方差分析 📉📈

**功能**：分析主成分分析的效果，决定是否需要降维

**使用方法**：
```python
preprocessor.plot_pca_variance(X, max_components=20, figsize=(12, 5))
```

**解读**：
- 左图：每个主成分解释的方差
- 右图：累积方差解释
- 虚线显示90%和95%方差阈值
- **建议**：如果前几个主成分就能解释大部分方差，可以考虑使用PCA

## 完整工作流程示例

```python
# 1. 加载和预处理数据
preprocessor = MarchMadnessPreprocessor()
preprocessor.load_data()
preprocessor.clean_data()
season_stats = preprocessor.create_season_stats()
features_df = preprocessor.create_matchup_features(season_stats, preprocessor.data.get('tourney_seeds'))
X, y = preprocessor.prepare_features_for_modeling(features_df)

# 2. 快速查看所有可视化
preprocessor.visualize_all(X, y)

# 3. 深入分析特定特征（可选）
# 查看相关性
preprocessor.plot_feature_correlation(X)

# 查看重要性（使用随机森林，更准确）
importance = preprocessor.plot_feature_importance(X, y, method='random_forest')

# 根据重要性筛选特征
top_features = importance.head(10).index.tolist()
X_selected = X[top_features]

# 4. 如果特征太多，考虑PCA
preprocessor.plot_pca_variance(X)
X_pca, pca_model = preprocessor.apply_pca(X, variance_threshold=0.95)
```

## 特征选择建议

基于可视化结果：

1. **删除高度相关的特征**（相关性 > 0.7）
   - 保留重要性更高的那个

2. **保留重要性高的特征**
   - 随机森林重要性 > 0.05
   - 或相关性 > 0.3

3. **检查特征分布**
   - 如果分布差异明显，保留该特征
   - 如果分布几乎相同，考虑删除

4. **考虑PCA**
   - 如果特征数 > 20
   - 且前几个主成分能解释 > 90% 方差

## 示例输出解读

### 特征重要性排名示例：
```
Top 15 Most Important Features:
   1. ScoreDiff_Diff          : 0.2341  ← 最重要！
   2. WinPct_Diff             : 0.1892
   3. Team1_AvgScoreDiff      : 0.1567
   4. Seed_Diff               : 0.1234
   ...
```

### 相关性分析输出：
```
⚠ Found 3 highly correlated pairs (|r| > 0.7):
  PointsFor_Diff <-> AvgPointsFor_Diff: 0.856
  PointsAgainst_Diff <-> AvgPointsAgainst_Diff: 0.892
```

**建议**：删除 AvgPointsFor_Diff 和 AvgPointsAgainst_Diff，保留 PointsFor_Diff 和 PointsAgainst_Diff

## 常见问题

**Q: 可视化不显示？**
A: 确保安装了 matplotlib 和 seaborn：`pip install matplotlib seaborn`

**Q: 图表显示中文乱码？**
A: 代码使用英文标签，避免中文乱码问题

**Q: 如何保存图表？**
A: 在 Jupyter notebook 中，图表会自动显示。如需保存，可以：
```python
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
```

**Q: 可视化太慢？**
A: 可以：
- 减少 `top_n` 参数
- 使用 `correlation` 方法而不是 `random_forest`
- 只运行需要的特定可视化

## 下一步

完成可视化后：
1. 根据结果选择最重要的特征
2. 删除冗余特征
3. 如果需要，应用 PCA
4. 开始训练模型

祝你预测成功！🏀


