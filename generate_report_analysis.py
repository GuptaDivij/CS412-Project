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

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11

from ml_pipeline import MarchMadnessPreprocessor

def create_output_dir():
    output_dir = Path('report_analysis')
    output_dir.mkdir(exist_ok=True)
    return output_dir

def ablation_study(preprocessor, features_df, output_dir):
    print("Running ablation study...")
    X_tr, X_val, y_tr, y_val, meta_tr, meta_val, feat_cols = preprocessor.split_train_test(
        features_df,
        method="season",
        train_seasons=list(range(1985, 2021)),
        val_seasons=[2021, 2022, 2023]
    )
    
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
        
        model = xgb.XGBClassifier(learning_rate=0.05, max_depth=3, n_estimators=50, 
                                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        model.fit(X_tr[features], y_tr)
        y_pred = model.predict_proba(X_val[features])[:, 1]
        results[group_name] = {
            'n_features': len(features),
            'log_loss': log_loss(y_val, y_pred),
            'accuracy': accuracy_score(y_val, (y_pred >= 0.5).astype(int)),
            'auc': roc_auc_score(y_val, y_pred)
        }
    
    pd.DataFrame(results).T.to_csv(output_dir / 'ablation_study_results.csv')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = range(len(results))
    labels = list(results.keys())
    
    axes[0].bar(x, [r['log_loss'] for r in results.values()])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_ylabel('Log Loss (lower is better)')
    axes[0].set_title('Ablation Study: Log Loss')
    axes[0].axhline(y=results['All Features']['log_loss'], color='r', linestyle='--', label='Baseline')
    axes[0].legend()
    
    axes[1].bar(x, [r['accuracy'] for r in results.values()])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylabel('Accuracy (higher is better)')
    axes[1].set_title('Ablation Study: Accuracy')
    axes[1].axhline(y=results['All Features']['accuracy'], color='r', linestyle='--', label='Baseline')
    axes[1].legend()
    
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
    
    return results

def parameter_study(preprocessor, features_df, output_dir):
    print("Running parameter study...")
    X_tr, X_val, y_tr, y_val, meta_tr, meta_val, feat_cols = preprocessor.split_train_test(
        features_df,
        method="season",
        train_seasons=list(range(1985, 2021)),
        val_seasons=[2021, 2022, 2023]
    )
    
    lr_results = []
    for lr in [0.01, 0.03, 0.05, 0.1, 0.2]:
        model = xgb.XGBClassifier(learning_rate=lr, max_depth=3, n_estimators=50, random_state=42, verbosity=0)
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_val)[:, 1]
        lr_results.append({'learning_rate': lr, 'log_loss': log_loss(y_val, y_pred)})
    
    depth_results = []
    for depth in [2, 3, 4, 5, 6, 7]:
        model = xgb.XGBClassifier(learning_rate=0.05, max_depth=depth, n_estimators=50, random_state=42, verbosity=0)
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_val)[:, 1]
        depth_results.append({'max_depth': depth, 'log_loss': log_loss(y_val, y_pred)})
    
    nestim_results = []
    for n_est in [10, 25, 50, 100, 150, 200]:
        model = xgb.XGBClassifier(learning_rate=0.05, max_depth=3, n_estimators=n_est, random_state=42, verbosity=0)
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_val)[:, 1]
        nestim_results.append({'n_estimators': n_est, 'log_loss': log_loss(y_val, y_pred)})
    
    pd.DataFrame(lr_results).to_csv(output_dir / 'param_study_learning_rate.csv', index=False)
    pd.DataFrame(depth_results).to_csv(output_dir / 'param_study_max_depth.csv', index=False)
    pd.DataFrame(nestim_results).to_csv(output_dir / 'param_study_n_estimators.csv', index=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    lr_df = pd.DataFrame(lr_results)
    axes[0].plot(lr_df['learning_rate'], lr_df['log_loss'], marker='o')
    axes[0].set_xlabel('Learning Rate')
    axes[0].set_ylabel('Log Loss')
    axes[0].set_title('Parameter Study: Learning Rate')
    axes[0].grid(True, alpha=0.3)
    
    depth_df = pd.DataFrame(depth_results)
    axes[1].plot(depth_df['max_depth'], depth_df['log_loss'], marker='o')
    axes[1].set_xlabel('Max Depth')
    axes[1].set_ylabel('Log Loss')
    axes[1].set_title('Parameter Study: Max Depth')
    axes[1].grid(True, alpha=0.3)
    
    nestim_df = pd.DataFrame(nestim_results)
    axes[2].plot(nestim_df['n_estimators'], nestim_df['log_loss'], marker='o')
    axes[2].set_xlabel('Number of Estimators')
    axes[2].set_ylabel('Log Loss')
    axes[2].set_title('Parameter Study: N Estimators')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_study.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return lr_results, depth_results, nestim_results

def baseline_comparisons(preprocessor, features_df, output_dir):
    print("Running baseline comparisons...")
    X_tr, X_val, y_tr, y_val, meta_tr, meta_val, feat_cols = preprocessor.split_train_test(
        features_df,
        method="season",
        train_seasons=list(range(1985, 2021)),
        val_seasons=[2021, 2022, 2023]
    )
    
    results = {}
    
    y_random = np.full(len(y_val), 0.5)
    results['Random'] = {
        'log_loss': log_loss(y_val, y_random),
        'accuracy': 0.5,
        'auc': 0.5
    }
    
    majority = y_tr.mode()[0]
    y_majority = np.full(len(y_val), majority)
    y_majority_proba = np.clip(y_majority, 0.01, 0.99)
    results['Majority Class'] = {
        'log_loss': log_loss(y_val, y_majority_proba),
        'accuracy': accuracy_score(y_val, y_majority),
        'auc': 0.5
    }
    
    seed_features = [c for c in feat_cols if 'Seed' in c]
    model_seed = LogisticRegression(max_iter=1000, random_state=42)
    model_seed.fit(X_tr[seed_features], y_tr)
    y_seed = model_seed.predict_proba(X_val[seed_features])[:, 1]
    results['Seed-Only'] = {
        'log_loss': log_loss(y_val, y_seed),
        'accuracy': accuracy_score(y_val, (y_seed >= 0.5).astype(int)),
        'auc': roc_auc_score(y_val, y_seed)
    }
    
    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_lr.fit(X_tr, y_tr)
    y_lr = model_lr.predict_proba(X_val)[:, 1]
    results['Logistic Regression'] = {
        'log_loss': log_loss(y_val, y_lr),
        'accuracy': accuracy_score(y_val, (y_lr >= 0.5).astype(int)),
        'auc': roc_auc_score(y_val, y_lr)
    }
    
    model_xgb = xgb.XGBClassifier(learning_rate=0.05, max_depth=3, n_estimators=50,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    model_xgb.fit(X_tr, y_tr)
    y_xgb = model_xgb.predict_proba(X_val)[:, 1]
    results['XGBoost (Ours)'] = {
        'log_loss': log_loss(y_val, y_xgb),
        'accuracy': accuracy_score(y_val, (y_xgb >= 0.5).astype(int)),
        'auc': roc_auc_score(y_val, y_xgb)
    }
    
    model_lgb = lgb.LGBMClassifier(learning_rate=0.05, max_depth=3, n_estimators=50,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
    model_lgb.fit(X_tr, y_tr)
    y_lgb = model_lgb.predict_proba(X_val)[:, 1]
    results['LightGBM (Ours)'] = {
        'log_loss': log_loss(y_val, y_lgb),
        'accuracy': accuracy_score(y_val, (y_lgb >= 0.5).astype(int)),
        'auc': roc_auc_score(y_val, y_lgb)
    }
    
    pd.DataFrame(results).T.to_csv(output_dir / 'baseline_comparison.csv')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    models = list(results.keys())
    x = range(len(models))
    
    axes[0].bar(x, [r['log_loss'] for r in results.values()])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylabel('Log Loss (lower is better)')
    axes[0].set_title('Model Comparison: Log Loss')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(x, [r['accuracy'] for r in results.values()])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Model Comparison: Accuracy')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(x, [r['auc'] for r in results.values()])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45, ha='right')
    axes[2].set_ylabel('AUC')
    axes[2].set_title('Model Comparison: AUC')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def cross_validation_analysis(preprocessor, features_df, output_dir):
    print("Running cross-validation...")
    cv_data = features_df[features_df['Season'] <= 2020].copy()
    feature_cols = [c for c in cv_data.columns if c not in ['Season', 'Team1', 'Team2', 'Outcome']]
    X = cv_data[feature_cols]
    y = cv_data['Outcome']
    
    tscv = TimeSeriesSplit(n_splits=5)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost': xgb.XGBClassifier(learning_rate=0.05, max_depth=3, n_estimators=50, 
                                     random_state=42, verbosity=0),
        'LightGBM': lgb.LGBMClassifier(learning_rate=0.05, max_depth=3, n_estimators=50, 
                                      random_state=42, verbosity=-1)
    }
    
    cv_results = {name: [] for name in models}
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]
            cv_results[name].append(log_loss(y_val, y_pred))
    
    cv_stats = {}
    for name, scores in cv_results.items():
        cv_stats[name] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }
    
    pd.DataFrame(cv_results).to_csv(output_dir / 'cross_validation_scores.csv', index=False)
    pd.DataFrame(cv_stats).T.to_csv(output_dir / 'cross_validation_stats.csv')
    
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
    
    return cv_results, cv_stats

def main():
    print("Generating report analysis...")
    output_dir = create_output_dir()
    
    preprocessor = MarchMadnessPreprocessor('march-machine-learning-mania-2025')
    preprocessor.load_data()
    cleaned = preprocessor.clean_data(verbose=False)
    preprocessor.data = preprocessor.cleaned
    season_stats = preprocessor.create_season_stats()
    features_df = preprocessor.create_features()
    
    ablation_results = ablation_study(preprocessor, features_df, output_dir)
    param_results = parameter_study(preprocessor, features_df, output_dir)
    baseline_results = baseline_comparisons(preprocessor, features_df, output_dir)
    cv_results, cv_stats = cross_validation_analysis(preprocessor, features_df, output_dir)
    
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ablation_study': ablation_results,
        'baseline_comparison': baseline_results,
        'cross_validation_stats': cv_stats,
    }
    
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=convert)
    
    print(f"Analysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()

