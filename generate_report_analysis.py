"""
Generate comprehensive analysis for final report
Includes: ablation study, parameter study, baseline comparisons
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import json
from pathlib import Path
from datetime import datetime

# Set style for publication-quality figures
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11

from ml_pipeline import MarchMadnessPreprocessor

def create_output_dir():
    """Create directory for report figures and results"""
    report_dir = Path('report_analysis')
    report_dir.mkdir(exist_ok=True)
    return report_dir

# ============================================================================
# 1. ABLATION STUDY - Remove features one at a time
# ============================================================================

def ablation_study(preprocessor, features_df, output_dir):
    """
    Test impact of removing different feature groups
    """
    print("\n" + "="*70)
    print("ABLATION STUDY")
    print("="*70)
    
    # Prepare data
    X_tr, X_val, y_tr, y_val, meta_tr, meta_val, feat_cols = preprocessor.split_train_test(
        features_df,
        method="season",
        train_seasons=list(range(1985, 2021)),
        val_seasons=[2021, 2022, 2023]
    )
    
    # Define feature groups
    feature_groups = {
        'All Features': feat_cols,
        'Without Seeds': [c for c in feat_cols if 'Seed' not in c],
        'Without WinPct': [c for c in feat_cols if 'WinPct' not in c],
        'Without Scoring': [c for c in feat_cols if 'Points' not in c and 'ScoreDiff' not in c],
        'Seeds Only': [c for c in feat_cols if 'Seed' in c],
        'Stats Only': [c for c in feat_cols if 'Seed' not in c],
    }
    
    results = {}
    
    for group_name, features in feature_groups.items():
        if len(features) == 0:
            continue
            
        print(f"\n[{group_name}] Using {len(features)} features")
        
        # Train XGBoost with this feature set
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            learning_rate=0.05,
            max_depth=3,
            n_estimators=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        
        model.fit(X_tr[features], y_tr)
        
        y_pred = model.predict_proba(X_val[features])[:, 1]
        ll = log_loss(y_val, y_pred)
        acc = accuracy_score(y_val, (y_pred >= 0.5).astype(int))
        auc = roc_auc_score(y_val, y_pred)
        
        results[group_name] = {
            'n_features': len(features),
            'log_loss': ll,
            'accuracy': acc,
            'auc': auc
        }
        
        print(f"  Log Loss: {ll:.4f} | Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_dir / 'ablation_study_results.csv')
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    x = range(len(results))
    labels = list(results.keys())
    
    # Log Loss
    axes[0].bar(x, [r['log_loss'] for r in results.values()])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_ylabel('Log Loss (lower is better)')
    axes[0].set_title('Ablation Study: Log Loss')
    axes[0].axhline(y=results['All Features']['log_loss'], color='r', linestyle='--', label='Baseline')
    axes[0].legend()
    
    # Accuracy
    axes[1].bar(x, [r['accuracy'] for r in results.values()])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylabel('Accuracy (higher is better)')
    axes[1].set_title('Ablation Study: Accuracy')
    axes[1].axhline(y=results['All Features']['accuracy'], color='r', linestyle='--', label='Baseline')
    axes[1].legend()
    
    # AUC
    axes[2].bar(x, [r['auc'] for r in results.values()])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha='right')
    axes[2].set_ylabel('AUC (higher is better)')
    axes[2].set_title('Ablation Study: AUC')
    axes[2].axhline(y=results['All Features']['auc'], color='r', linestyle='--', label='Baseline')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_study.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[OK] Ablation study saved to {output_dir}")
    return results

# ============================================================================
# 2. PARAMETER STUDY - Vary hyperparameters
# ============================================================================

def parameter_study(preprocessor, features_df, output_dir):
    """
    Study effect of different hyperparameters
    """
    print("\n" + "="*70)
    print("PARAMETER STUDY")
    print("="*70)
    
    # Prepare data
    X_tr, X_val, y_tr, y_val, meta_tr, meta_val, feat_cols = preprocessor.split_train_test(
        features_df,
        method="season",
        train_seasons=list(range(1985, 2021)),
        val_seasons=[2021, 2022, 2023]
    )
    
    # Study 1: Learning Rate
    print("\n[1/3] Studying Learning Rate...")
    learning_rates = [0.01, 0.03, 0.05, 0.1, 0.2]
    lr_results = []
    
    for lr in learning_rates:
        model = xgb.XGBClassifier(
            learning_rate=lr,
            max_depth=3,
            n_estimators=50,
            random_state=42,
            verbosity=0
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, y_pred)
        lr_results.append({'learning_rate': lr, 'log_loss': ll})
        print(f"  LR={lr:.2f}: {ll:.4f}")
    
    # Study 2: Max Depth
    print("\n[2/3] Studying Max Depth...")
    depths = [2, 3, 4, 5, 6, 7]
    depth_results = []
    
    for depth in depths:
        model = xgb.XGBClassifier(
            learning_rate=0.05,
            max_depth=depth,
            n_estimators=50,
            random_state=42,
            verbosity=0
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, y_pred)
        depth_results.append({'max_depth': depth, 'log_loss': ll})
        print(f"  Depth={depth}: {ll:.4f}")
    
    # Study 3: Number of Estimators
    print("\n[3/3] Studying Number of Estimators...")
    n_estimators_list = [10, 25, 50, 100, 150, 200]
    nestim_results = []
    
    for n_est in n_estimators_list:
        model = xgb.XGBClassifier(
            learning_rate=0.05,
            max_depth=3,
            n_estimators=n_est,
            random_state=42,
            verbosity=0
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, y_pred)
        nestim_results.append({'n_estimators': n_est, 'log_loss': ll})
        print(f"  N={n_est}: {ll:.4f}")
    
    # Save results
    pd.DataFrame(lr_results).to_csv(output_dir / 'param_study_learning_rate.csv', index=False)
    pd.DataFrame(depth_results).to_csv(output_dir / 'param_study_max_depth.csv', index=False)
    pd.DataFrame(nestim_results).to_csv(output_dir / 'param_study_n_estimators.csv', index=False)
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Learning Rate
    lr_df = pd.DataFrame(lr_results)
    axes[0].plot(lr_df['learning_rate'], lr_df['log_loss'], marker='o')
    axes[0].set_xlabel('Learning Rate')
    axes[0].set_ylabel('Log Loss')
    axes[0].set_title('Parameter Study: Learning Rate')
    axes[0].grid(True, alpha=0.3)
    
    # Max Depth
    depth_df = pd.DataFrame(depth_results)
    axes[1].plot(depth_df['max_depth'], depth_df['log_loss'], marker='o')
    axes[1].set_xlabel('Max Depth')
    axes[1].set_ylabel('Log Loss')
    axes[1].set_title('Parameter Study: Max Depth')
    axes[1].grid(True, alpha=0.3)
    
    # N Estimators
    nestim_df = pd.DataFrame(nestim_results)
    axes[2].plot(nestim_df['n_estimators'], nestim_df['log_loss'], marker='o')
    axes[2].set_xlabel('Number of Estimators')
    axes[2].set_ylabel('Log Loss')
    axes[2].set_title('Parameter Study: N Estimators')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_study.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[OK] Parameter study saved to {output_dir}")
    return lr_results, depth_results, nestim_results

# ============================================================================
# 3. BASELINE COMPARISONS
# ============================================================================

def baseline_comparisons(preprocessor, features_df, output_dir):
    """
    Compare against multiple baselines
    """
    print("\n" + "="*70)
    print("BASELINE COMPARISONS")
    print("="*70)
    
    # Prepare data
    X_tr, X_val, y_tr, y_val, meta_tr, meta_val, feat_cols = preprocessor.split_train_test(
        features_df,
        method="season",
        train_seasons=list(range(1985, 2021)),
        val_seasons=[2021, 2022, 2023]
    )
    
    results = {}
    
    # Baseline 1: Random (always predict 0.5)
    print("\n[1/6] Random Baseline...")
    y_random = np.full(len(y_val), 0.5)
    results['Random'] = {
        'log_loss': log_loss(y_val, y_random),
        'accuracy': 0.5,
        'auc': 0.5
    }
    print(f"  Log Loss: {results['Random']['log_loss']:.4f}")
    
    # Baseline 2: Majority class
    print("\n[2/6] Majority Class Baseline...")
    majority = y_tr.mode()[0]
    y_majority = np.full(len(y_val), majority)
    # Clip for log_loss (can't be exactly 0 or 1)
    y_majority_proba = np.clip(y_majority, 0.01, 0.99)
    results['Majority Class'] = {
        'log_loss': log_loss(y_val, y_majority_proba),
        'accuracy': accuracy_score(y_val, y_majority),
        'auc': 0.5
    }
    print(f"  Log Loss: {results['Majority Class']['log_loss']:.4f}")
    
    # Baseline 3: Seed-only model
    print("\n[3/6] Seed-Only Baseline...")
    seed_features = [c for c in feat_cols if 'Seed' in c]
    model_seed = LogisticRegression(max_iter=1000, random_state=42)
    model_seed.fit(X_tr[seed_features], y_tr)
    y_seed = model_seed.predict_proba(X_val[seed_features])[:, 1]
    results['Seed-Only'] = {
        'log_loss': log_loss(y_val, y_seed),
        'accuracy': accuracy_score(y_val, (y_seed >= 0.5).astype(int)),
        'auc': roc_auc_score(y_val, y_seed)
    }
    print(f"  Log Loss: {results['Seed-Only']['log_loss']:.4f}")
    
    # Baseline 4: Logistic Regression (full)
    print("\n[4/6] Logistic Regression (full features)...")
    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_lr.fit(X_tr, y_tr)
    y_lr = model_lr.predict_proba(X_val)[:, 1]
    results['Logistic Regression'] = {
        'log_loss': log_loss(y_val, y_lr),
        'accuracy': accuracy_score(y_val, (y_lr >= 0.5).astype(int)),
        'auc': roc_auc_score(y_val, y_lr)
    }
    print(f"  Log Loss: {results['Logistic Regression']['log_loss']:.4f}")
    
    # Our Model 5: XGBoost
    print("\n[5/6] XGBoost (Our Model)...")
    model_xgb = xgb.XGBClassifier(
        learning_rate=0.05,
        max_depth=3,
        n_estimators=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model_xgb.fit(X_tr, y_tr)
    y_xgb = model_xgb.predict_proba(X_val)[:, 1]
    results['XGBoost (Ours)'] = {
        'log_loss': log_loss(y_val, y_xgb),
        'accuracy': accuracy_score(y_val, (y_xgb >= 0.5).astype(int)),
        'auc': roc_auc_score(y_val, y_xgb)
    }
    print(f"  Log Loss: {results['XGBoost (Ours)']['log_loss']:.4f}")
    
    # Our Model 6: LightGBM
    print("\n[6/6] LightGBM (Our Model)...")
    model_lgb = lgb.LGBMClassifier(
        learning_rate=0.05,
        max_depth=3,
        n_estimators=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    model_lgb.fit(X_tr, y_tr)
    y_lgb = model_lgb.predict_proba(X_val)[:, 1]
    results['LightGBM (Ours)'] = {
        'log_loss': log_loss(y_val, y_lgb),
        'accuracy': accuracy_score(y_val, (y_lgb >= 0.5).astype(int)),
        'auc': roc_auc_score(y_val, y_lgb)
    }
    print(f"  Log Loss: {results['LightGBM (Ours)']['log_loss']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_dir / 'baseline_comparison.csv')
    
    # Create comparison table
    print("\n" + "-"*70)
    print(f"{'Model':<25} {'Log Loss':<12} {'Accuracy':<12} {'AUC':<12}")
    print("-"*70)
    for name, res in results.items():
        print(f"{name:<25} {res['log_loss']:<12.4f} {res['accuracy']:<12.4f} {res['auc']:<12.4f}")
    print("-"*70)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = list(results.keys())
    x = range(len(models))
    
    # Log Loss
    axes[0].bar(x, [r['log_loss'] for r in results.values()])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylabel('Log Loss (lower is better)')
    axes[0].set_title('Model Comparison: Log Loss')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Accuracy
    axes[1].bar(x, [r['accuracy'] for r in results.values()])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Model Comparison: Accuracy')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # AUC
    axes[2].bar(x, [r['auc'] for r in results.values()])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45, ha='right')
    axes[2].set_ylabel('AUC')
    axes[2].set_title('Model Comparison: AUC')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[OK] Baseline comparison saved to {output_dir}")
    return results

# ============================================================================
# 4. CROSS-VALIDATION ANALYSIS
# ============================================================================

def cross_validation_analysis(preprocessor, features_df, output_dir):
    """
    Perform temporal cross-validation
    """
    print("\n" + "="*70)
    print("CROSS-VALIDATION ANALYSIS")
    print("="*70)
    
    # Prepare full dataset (exclude most recent years for CV)
    cv_data = features_df[features_df['Season'] <= 2020].copy()
    
    feature_cols = [c for c in cv_data.columns if c not in ['Season', 'Team1', 'Team2', 'Outcome']]
    X = cv_data[feature_cols]
    y = cv_data['Outcome']
    
    # Time series split (respects temporal order)
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost': xgb.XGBClassifier(learning_rate=0.05, max_depth=3, n_estimators=50, random_state=42, verbosity=0),
        'LightGBM': lgb.LGBMClassifier(learning_rate=0.05, max_depth=3, n_estimators=50, random_state=42, verbosity=-1)
    }
    
    cv_results = {name: [] for name in models}
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\n[Fold {fold}/5]")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            ll = log_loss(y_val, y_pred)
            cv_results[name].append(ll)
            print(f"  {name}: {ll:.4f}")
    
    # Calculate statistics
    cv_stats = {}
    for name, scores in cv_results.items():
        cv_stats[name] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }
    
    # Save results
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(output_dir / 'cross_validation_scores.csv', index=False)
    
    stats_df = pd.DataFrame(cv_stats).T
    stats_df.to_csv(output_dir / 'cross_validation_stats.csv')
    
    # Print summary
    print("\n" + "-"*70)
    print(f"{'Model':<25} {'Mean ± Std':<20} {'Min':<12} {'Max':<12}")
    print("-"*70)
    for name, stats in cv_stats.items():
        print(f"{name:<25} {stats['mean']:.4f} ± {stats['std']:.4f}   {stats['min']:<12.4f} {stats['max']:<12.4f}")
    print("-"*70)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = range(len(models))
    for i, (name, scores) in enumerate(cv_results.items()):
        ax.boxplot([scores], positions=[i], widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
        ax.scatter([i]*len(scores), scores, alpha=0.5, color='red')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(models.keys(), rotation=15, ha='right')
    ax.set_ylabel('Log Loss')
    ax.set_title('5-Fold Temporal Cross-Validation Results')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[OK] Cross-validation analysis saved to {output_dir}")
    return cv_results, cv_stats

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all analyses for final report"""
    print("="*70)
    print("GENERATING COMPREHENSIVE ANALYSIS FOR FINAL REPORT")
    print("="*70)
    
    # Create output directory
    output_dir = create_output_dir()
    
    # Load and prepare data
    print("\n[1/6] Loading data...")
    preprocessor = MarchMadnessPreprocessor('march-machine-learning-mania-2025')
    preprocessor.load_data()
    
    print("\n[2/6] Cleaning data...")
    cleaned = preprocessor.clean_data(verbose=False)
    preprocessor.data = preprocessor.cleaned
    
    print("\n[3/6] Creating features...")
    season_stats = preprocessor.create_season_stats()
    features_df = preprocessor.create_features()
    print(f"  Features shape: {features_df.shape}")
    
    # Run analyses
    print("\n[4/6] Running ablation study...")
    ablation_results = ablation_study(preprocessor, features_df, output_dir)
    
    print("\n[5/6] Running parameter study...")
    param_results = parameter_study(preprocessor, features_df, output_dir)
    
    print("\n[6/6] Running baseline comparisons...")
    baseline_results = baseline_comparisons(preprocessor, features_df, output_dir)
    
    print("\n[7/7] Running cross-validation...")
    cv_results, cv_stats = cross_validation_analysis(preprocessor, features_df, output_dir)
    
    # Save summary
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ablation_study': ablation_results,
        'baseline_comparison': baseline_results,
        'cross_validation_stats': cv_stats,
    }
    
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(summary, f, indent=2, default=convert)
    
    print("\n" + "="*70)
    print("[SUCCESS] All analyses complete!")
    print(f"Results saved to: {output_dir.absolute()}")
    print("="*70)
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*')):
        print(f"  - {file.name}")
    print("\nUse these results and figures in your final report!")

if __name__ == "__main__":
    main()

