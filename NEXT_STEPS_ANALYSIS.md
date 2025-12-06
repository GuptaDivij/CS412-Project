# 📋 Pre-Model-Improvement Checklist

**Date**: December 6, 2025  
**Current Status**: Logistic Regression (0.6894 log loss) outperforming tree models

---

## 🚨 CRITICAL ISSUES (Must Fix Before Model Tuning)

### 1. ⚠️ **Validation Set Too Small** - HIGH PRIORITY
**Problem**: Only 67 games from 2023 is statistically insufficient
- Random variance heavily impacts metrics
- Cannot reliably compare models
- Results may not generalize

**Impact**: Your current metrics are unreliable ❌

**Solutions**:
```python
# Option A: Use multiple validation years
train_seasons=list(range(1985, 2021))  # Train: 1985-2020
val_seasons=[2021, 2022, 2023]          # Val: 2021-2023 (~200 games)

# Option B: Implement k-fold cross-validation
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

**Priority**: 🔴 Fix IMMEDIATELY

---

### 2. ⚠️ **Tree Models Underperforming Linear Model** - HIGH PRIORITY
**Problem**: XGBoost/LightGBM worse than Logistic Regression is a RED FLAG
- Suggests overfitting on small validation set
- Feature engineering may not be suitable for tree models
- Hyperparameters might be completely wrong

**Evidence**:
- XGBoost: 0.7559 log loss (9.6% worse than baseline)
- LightGBM: 0.7615 log loss (10.5% worse than baseline)
- Both have ~48% accuracy (worse than random!)

**Root Causes**:
1. **Overfitting**: Small validation set + complex models = overfitting
2. **Feature Scaling**: Tree models don't need scaling, but current features might have issues
3. **Label Encoding**: Check if `Outcome` is correctly encoded
4. **Hyperparameters**: Default params (100 estimators, depth=5) may not fit this problem

**Solutions**:
```python
# A) Check for overfitting - compare train vs val performance
train_pred = model.predict_proba(X_train)[:, 1]
train_loss = log_loss(y_train, train_pred)
val_loss = log_loss(y_val, val_pred)
print(f"Train: {train_loss:.4f}, Val: {val_loss:.4f}")
# If train << val, you're overfitting

# B) Start with simpler models
xgb_model = xgb.XGBClassifier(
    max_depth=3,           # Reduce from 5
    n_estimators=50,       # Reduce from 100
    learning_rate=0.05,    # Reduce from 0.1
    subsample=0.8,         # Add regularization
    colsample_bytree=0.8   # Add regularization
)

# C) Use early stopping with larger validation set
```

**Priority**: 🔴 Investigate IMMEDIATELY

---

### 3. ⚠️ **Seeds Have Low Feature Importance** - MEDIUM PRIORITY
**Problem**: In March Madness, seeds are HIGHLY predictive, but your models rank them low
- XGBoost: `Seed_Diff` is 14th out of 17 features (only 4.98% importance)
- LightGBM: `Seed_Diff` is 14th out of 17 features (43 splits vs 124 for top feature)
- **This is WRONG** - seeds should be top 3 features

**Why This Matters**:
A simple seed-based baseline (e.g., "lower seed always wins") typically achieves:
- ~65-70% accuracy
- ~0.55-0.60 log loss

Your models aren't learning this basic pattern.

**Possible Causes**:
1. **Feature Representation**: `Seed_Diff` might not be the best representation
   - Try: `Team1_Seed`, `Team2_Seed` separately
   - Try: `Seed_Product`, `Seed_Ratio`, interaction terms

2. **Data Leakage Check**: Verify seeds are from BEFORE tournament (not after)
   ```python
   # Check: Are you getting seeds correctly?
   print(features_df[['Season', 'Team1', 'Team2', 'Team1_Seed', 'Team2_Seed', 'Outcome']].head(20))
   ```

3. **Correlation with Other Features**: Seeds might be redundant with `WinPct_Diff`
   - Check correlation matrix
   - If highly correlated, tree models might prefer other features

**Solution**:
```python
# Add seed-based features
features['Seed_Interaction'] = features['Team1_Seed'] * features['Team2_Seed']
features['Seed_Ratio'] = features['Team1_Seed'] / (features['Team2_Seed'] + 1e-6)
features['Is_Upset_Potential'] = (features['Seed_Diff'] >= 3).astype(int)

# Try a seed-only baseline model
from sklearn.linear_model import LogisticRegression
seed_only = LogisticRegression()
seed_only.fit(X_train[['Team1_Seed', 'Team2_Seed', 'Seed_Diff']], y_train)
seed_pred = seed_only.predict_proba(X_val[['Team1_Seed', 'Team2_Seed', 'Seed_Diff']])[:, 1]
print(f"Seed-only log loss: {log_loss(y_val, seed_pred):.4f}")
# Should be < 0.60 if seeds are working properly
```

**Priority**: 🟡 Investigate SOON

---

### 4. ⚠️ **Missing Important Features** - MEDIUM PRIORITY
**Current Features (17 total)**:
- Season stats: Win%, Points For/Against, Score Diff
- Seeds: Team1_Seed, Team2_Seed, Seed_Diff
- Games played

**Missing High-Value Features**:
1. **Momentum/Recency**:
   - Last 10 games win%
   - Win streak
   - Recent performance trend

2. **Strength of Schedule**:
   - Opponent win% (SOS)
   - Conference strength (using `MTeamConferences.csv`)

3. **Advanced Stats** (from `MRegularSeasonDetailedResults.csv` - you're loading it but not using it!):
   - Field Goal %, 3PT%, Free Throw %
   - Rebounds, Assists, Turnovers
   - Tempo (possessions per game)
   - Offensive/Defensive efficiency

4. **Tournament-Specific**:
   - Conference tournament performance (from `MConferenceTourneyGames.csv`)
   - Previous tournament experience (how far team went in past years)

5. **Ranking Systems** (from `MMasseyOrdinals.csv` - 5.5M rows!):
   - You're loading this but not using it
   - Features like RPI, KenPom, BPI, Sagarin ratings are VERY predictive

**Impact**: Adding these could improve log loss by 0.05-0.10

**Priority**: 🟡 Add BEFORE hyperparameter tuning

---

### 5. ⚠️ **No Cross-Validation** - MEDIUM PRIORITY
**Problem**: Training once on full train set doesn't detect overfitting
- Can't assess model stability
- Can't reliably compare models
- Hyperparameter tuning will be unreliable

**Solution**:
```python
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# Use TimeSeriesSplit for temporal data
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(
    model, X_train, y_train, 
    cv=tscv, 
    scoring='neg_log_loss'
)
print(f"CV Log Loss: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

**Priority**: 🟡 Add BEFORE hyperparameter tuning

---

### 6. ⚠️ **No Train Set Evaluation** - LOW PRIORITY
**Problem**: Not checking train performance = can't diagnose overfitting vs underfitting

**Solution**:
```python
# Add to train_and_evaluate_models()
y_train_pred = model.predict_proba(X_train)[:, 1]
train_loss = log_loss(y_train, y_train_pred)
print(f"  Train Log Loss: {train_loss:.4f}")

# Interpret:
# - train_loss << val_loss → Overfitting
# - train_loss ≈ val_loss (both high) → Underfitting
# - train_loss ≈ val_loss (both low) → Good fit ✓
```

**Priority**: 🟢 Nice to have for diagnostics

---

## 📊 RECOMMENDED ACTION PLAN

### Phase 1: Fix Data Issues (Do First!) 🔴
**Estimated Time**: 2-3 hours

1. **Expand validation set** to 2021-2023 (~200-300 games)
   - Edit line 824-826 in `ml_pipeline.py`
   
2. **Add train/val diagnostics**:
   - Check train performance to detect overfitting
   - Print confusion matrices
   - Analyze misclassifications

3. **Verify data correctness**:
   - Check seed values are reasonable (1-16)
   - Verify `Outcome` labels are correct
   - Check for data leakage (no future info in features)

**Code Changes**:
```python
# In main(), line 821-826, change to:
X_tr, X_val, y_tr, y_val, meta_tr, meta_val, feat_cols = preprocessor.split_train_test(
    features_df,
    method="season",
    train_seasons=list(range(1985, 2021)),  # Changed from 2023
    val_seasons=[2021, 2022, 2023]          # Changed to multiple years
)
```

---

### Phase 2: Fix Model Issues (Do Second!) 🟡
**Estimated Time**: 1-2 hours

1. **Start with simpler tree models**:
   - Reduce depth to 3
   - Reduce estimators to 50
   - Add regularization

2. **Implement cross-validation**:
   - Use `TimeSeriesSplit(n_splits=5)`
   - Report mean ± std of CV scores

3. **Create seed baseline**:
   - Train model on seeds only
   - This should get ~0.55-0.60 log loss
   - If not, seeds are broken

---

### Phase 3: Feature Engineering (Do Third!) 🟢
**Estimated Time**: 3-5 hours

1. **Add momentum features**:
   - Last 10 games performance
   - Win streaks

2. **Add advanced stats** from detailed results:
   - FG%, 3PT%, FT%
   - Rebounds, Assists, Turnovers

3. **Add Massey rankings**:
   - Use final rankings before tournament
   - Add RPI, KenPom, etc.

4. **Add strength of schedule**:
   - Conference strength
   - Opponent quality

---

### Phase 4: Model Tuning (Do Last!) ⚪
**Only after Phases 1-3 are complete**

1. Hyperparameter search (Grid/Random/Bayesian)
2. Ensemble methods
3. Advanced techniques (stacking, calibration)

---

## 🎯 Expected Results After Fixes

### After Phase 1 (Data Fixes):
- Validation set: ~200-300 games (more reliable)
- Can properly diagnose overfitting
- **Target**: Confirm LR baseline ~0.65-0.68 on larger val set

### After Phase 2 (Model Fixes):
- Tree models should match or beat LR baseline
- **Target**: XGBoost/LightGBM ~0.62-0.65 log loss

### After Phase 3 (Feature Engineering):
- Seeds should be top 3 features
- **Target**: Best model ~0.55-0.60 log loss

### After Phase 4 (Hyperparameter Tuning):
- Optimized for competition
- **Target**: Best model ~0.52-0.57 log loss (competitive with Kaggle)

---

## 📚 Quick Reference: March Madness Benchmarks

| Method | Expected Log Loss | Expected Accuracy |
|--------|------------------|-------------------|
| Random guessing | 0.693 | 50% |
| Seed-only baseline | 0.55-0.60 | 65-70% |
| Basic ML (current) | 0.60-0.65 | 62-68% |
| Good ML (w/ features) | 0.55-0.60 | 68-72% |
| Advanced ML (tuned) | 0.52-0.57 | 70-75% |
| Top Kaggle submissions | 0.48-0.52 | 72-77% |

**Your Current Performance**:
- Logistic Regression: 0.6894 (between random and seed baseline) ⚠️
- Tree models: 0.75+ (worse than random) 🚫

---

## 💡 Key Insights

1. **Your friend's EDA addition is great** - use those visualizations to validate your fixes
2. **Don't tune hyperparameters yet** - fix fundamental issues first
3. **Seeds are the most important feature** - if they're not, something is wrong
4. **Small validation set = unreliable metrics** - expand it first
5. **Tree models failing = likely overfitting** - simplify before complexifying

---

## ✅ Immediate Next Steps (Priority Order)

1. [ ] Run diagnostics: Check train vs val performance
2. [ ] Expand validation set to 2021-2023
3. [ ] Create seed-only baseline model
4. [ ] Simplify tree model hyperparameters
5. [ ] Add cross-validation
6. [ ] Analyze feature importance on new validation set
7. [ ] Add momentum and advanced stat features
8. [ ] Only then: hyperparameter tuning

---

**Bottom Line**: You have a solid foundation, but need to fix data/model issues before improving performance. Focus on Phases 1-2 first, which should take ~3-5 hours total.

