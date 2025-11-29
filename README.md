# NCAA March Madness 2025 Prediction

**Team:** Carlos (carlosd5), Divij Gupta (divij3), Rui Lin (rayl3), Sijie Mei (sijiem2)

Predicts win probabilities for NCAA Men's Basketball Tournament matchups using XGBoost and LightGBM models optimized with log loss.

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm
```

### 2. Run the Pipeline

```bash
python ml_pipeline.py
```

**Runtime:** 3-5 minutes

---

## 📊 What the Program Does

The pipeline executes 8 steps automatically:

1. **Loads data** - 22 CSV files with tournament and regular season data (1985-2024)
2. **Cleans data** - Removes missing values and duplicates
3. **Creates season stats** - Calculates win%, points, score margins per team
4. **Creates features** - Builds matchup comparisons with tournament seeds
5. **Splits data** - Train on 1985-2022, validate on 2023
6. **Applies PCA** - Reduces dimensionality (optional)
7. **Trains models** - Logistic Regression, XGBoost, LightGBM
8. **Saves results** - Exports predictions and metrics to `output/` folder

---

## 📁 Output Files

All results are saved to the `output/` directory with timestamps:

| File | Description |
|------|-------------|
| `predictions_*.csv` | Model predictions for all validation games |
| `metrics_*.json` | Performance metrics (log loss, accuracy, AUC) |
| `model_*.pkl` | Trained models (can reload for 2025 predictions) |
| `feature_importance_*.csv` | Feature rankings for tree models |
| `summary_report_*.txt` | Human-readable summary |

---

## 📈 Interpreting Results

### Console Output

At the end of the run, you'll see:

```
MODEL TRAINING & EVALUATION
------------------------------------------------------------
Model                Log Loss     Accuracy     AUC         
------------------------------------------------------------
Logistic Regression  0.6894       0.5672       0.6231      
XGBoost              0.7559       0.4776       0.4963      
LightGBM             0.7615       0.4776       0.5056      
------------------------------------------------------------
🏆 Best Model: Logistic Regression (Log Loss: 0.6894)
```

### Metrics Explained

**Log Loss** (lower is better)
- **< 0.60:** Excellent - competitive with Kaggle leaderboards
- **0.60-0.65:** Good baseline performance
- **0.65-0.70:** Acceptable but needs improvement
- **> 0.70:** Poor - worse than basic seed-only models

**Accuracy** (higher is better)
- **> 70%:** Strong predictive power
- **60-70%:** Moderate performance
- **50-60%:** Weak, close to random guessing
- **< 50%:** Worse than random (model has issues)

**AUC** (higher is better)
- **> 0.70:** Good discrimination between wins/losses
- **0.60-0.70:** Moderate discrimination
- **0.50-0.60:** Poor discrimination
- **0.50:** No better than random

### Typical Good Results

A well-performing model should achieve:
- Log Loss: **0.55-0.62**
- Accuracy: **65-72%**
- AUC: **0.70-0.78**

---

## 🎯 Using Saved Models

### Load a Model for 2025 Predictions

```python
import pickle
import pandas as pd

# Load the best model
with open('output/model_logistic_regression_20251123_152915.pkl', 'rb') as f:
    model = pickle.load(f)

# Create features for 2025 matchups (using your preprocessor)
# X_2025 = preprocessor.create_features_for_2025()

# Generate predictions
predictions_2025 = model.predict_proba(X_2025)[:, 1]
```

### Analyze Feature Importance

```python
import pandas as pd

feat_imp = pd.read_csv('output/feature_importance_xgboost_20251123_152915.csv')
print(feat_imp.head(10))  # Top 10 most important features
```

**Expected top features:**
1. `Seed_Diff` - Tournament seed difference (most important!)
2. `WinPct_Diff` - Win percentage difference
3. `ScoreDiff_Diff` - Score margin difference

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `FileNotFoundError` | Make sure you're in the project root directory |
| Poor model performance | Try adding more features or tuning hyperparameters |
| Encoding errors | Models and metrics still save; only summary_report affected |

---

## 📚 Project Details

- **Dataset:** Kaggle March Machine Learning Mania 2025 (22 tables)
- **Features:** 17 matchup-level features including seeds, win%, scoring
- **Models:** Logistic Regression (baseline), XGBoost, LightGBM
- **Evaluation:** Binary cross-entropy (log loss)
- **Validation:** Temporal split (train ≤2022, validate 2023)

---

## 📖 References

- IEEE (2021). Deep Similarity Learning for Sports Team Ranking
- Kocher & Hoblin (2018). Predictive Model for NCAA Men's Basketball
- Peng (2025). Machine Learning Models for NBA Game Prediction
- Tjärnhell et al. (2025). Evaluating ML Models for NCAA Division I Basketball

---

**Last Updated:** November 23, 2025  
**Course:** CS412 - Data Mining Principles (UIUC)
