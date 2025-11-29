# 安装指南

如果遇到 `No module named 'sklearn'` 或其他模块缺失错误，请按照以下步骤安装依赖。

## 快速安装

### 方法 1：使用 requirements.txt（推荐）

```bash
pip install -r requirements.txt
```

### 方法 2：单独安装

```bash
# 安装所有必需的包
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 方法 3：在 Jupyter Notebook 中安装

如果使用 Jupyter Notebook，可以在 notebook 的第一个 cell 中运行：

```python
!pip install pandas numpy scikit-learn matplotlib seaborn
```

## 验证安装

运行以下 Python 代码验证所有包都已安装：

```python
try:
    import pandas as pd
    import numpy as np
    import sklearn
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✓ 所有包已成功安装！")
    print(f"  pandas: {pd.__version__}")
    print(f"  numpy: {np.__version__}")
    print(f"  sklearn: {sklearn.__version__}")
    print(f"  matplotlib: {plt.matplotlib.__version__}")
    print(f"  seaborn: {sns.__version__}")
except ImportError as e:
    print(f"✗ 安装失败: {e}")
    print("请运行: pip install -r requirements.txt")
```

## 常见问题

### 问题 1: `No module named 'sklearn'`

**原因**: scikit-learn 未安装或安装在不同的 Python 环境中

**解决方法**:
```bash
pip install scikit-learn
```

或者使用 conda（如果使用 Anaconda）:
```bash
conda install scikit-learn
```

### 问题 2: 在 Jupyter Notebook 中仍然报错

**原因**: Jupyter Notebook 可能使用了不同的 Python 内核

**解决方法**:
1. 检查 Jupyter 使用的 Python 环境：
   ```python
   import sys
   print(sys.executable)
   ```

2. 在该环境中安装包：
   ```bash
   # 使用上面打印的路径
   /path/to/python -m pip install scikit-learn
   ```

3. 或者在 Jupyter 中直接安装：
   ```python
   !pip install scikit-learn
   ```

### 问题 3: 权限错误

**解决方法**:
```bash
# Windows
pip install --user scikit-learn

# Linux/Mac
pip install --user scikit-learn
```

### 问题 4: 使用虚拟环境

如果使用虚拟环境，确保先激活它：

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 然后安装包
pip install -r requirements.txt
```

## 完整依赖列表

当前项目需要以下 Python 包：

- `pandas>=1.5.0` - 数据处理
- `numpy>=1.20.0` - 数值计算
- `scikit-learn>=1.0.0` - 机器学习（PCA, Random Forest等）
- `matplotlib>=3.5.0` - 绘图
- `seaborn>=0.12.0` - 统计可视化

## 检查当前环境

运行以下命令查看已安装的包：

```bash
pip list
```

或者查看特定包：

```bash
pip show scikit-learn
```

## 更新包

如果遇到版本兼容问题，可以更新所有包：

```bash
pip install --upgrade pandas numpy scikit-learn matplotlib seaborn
```

## 使用 Anaconda

如果使用 Anaconda，可以使用 conda 安装：

```bash
conda install pandas numpy scikit-learn matplotlib seaborn
```

或者从 conda-forge 安装（通常更新）：

```bash
conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn
```

## 需要帮助？

如果仍然遇到问题：

1. 检查 Python 版本（推荐 3.8+）：
   ```bash
   python --version
   ```

2. 检查 pip 版本：
   ```bash
   pip --version
   ```

3. 更新 pip：
   ```bash
   python -m pip install --upgrade pip
   ```

4. 尝试重新安装：
   ```bash
   pip uninstall scikit-learn
   pip install scikit-learn
   ```


