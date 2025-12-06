"""
Quick Diagnostics Script for March Madness ML Pipeline
Run this to identify issues before improving model performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
import json

def run_diagnostics():
    print("="*70)
    print("MARCH MADNESS ML PIPELINE - DIAGNOSTIC REPORT")
    print("="*70)
    
    # Import the preprocessor
    from ml_pipeline import MarchMadnessPreprocessor
    
    # Load data
    print("\n[1/7] Loading data...")
    preprocessor = MarchMadnessPreprocessor('march-machine-learning-mania-2025')
    preprocessor.load_data()
    
    # Clean data
    print("\n[2/7] Cleaning data...")
    cleaned = preprocessor.clean_data(verbose=False)
    preprocessor.data = preprocessor.cleaned
    
    # Create features
    print("\n[3/7] Creating features...")
    season_stats = preprocessor.create_season_stats()
    features_df = preprocessor.create_features()
    
    print(f"[OK] Total games: {len(features_df)}")
    print(f"[OK] Seasons covered: {features_df['Season'].min()} - {features_df['Season'].max()}")
    print(f"[OK] Features created: {len([c for c in features_df.columns if c not in ['Season', 'Team1', 'Team2', 'Outcome']])}")
    
    # Check validation set size
    print("\n" + "="*70)
    print("[4/7] VALIDATION SET ANALYSIS")
    print("="*70)
    
    val_2023 = features_df[features_df['Season'] == 2023]
    val_2021_2023 = features_df[features_df['Season'].isin([2021, 2022, 2023])]
    
    print(f"\nCurrent approach (2023 only):")
    print(f"  Games: {len(val_2023)} [WARNING] TOO SMALL")
    print(f"  Seasons: 1")
    
    print(f"\nRecommended approach (2021-2023):")
    print(f"  Games: {len(val_2021_2023)} [OK] BETTER")
    print(f"  Seasons: 3")
    print(f"  Breakdown: {dict(val_2021_2023['Season'].value_counts().sort_index())}")
    
    if len(val_2023) < 100:
        print("\n[CRITICAL] Validation set too small (<100 games)")
        print("   -> Metrics are unreliable")
        print("   -> Recommendation: Use 2021-2023 for validation")
    
    # Check seed feature
    print("\n" + "="*70)
    print("[5/7] SEED FEATURE ANALYSIS")
    print("="*70)
    
    print("\nSeed value distribution:")
    print(f"  Team1_Seed range: {features_df['Team1_Seed'].min():.0f} - {features_df['Team1_Seed'].max():.0f}")
    print(f"  Team2_Seed range: {features_df['Team2_Seed'].min():.0f} - {features_df['Team2_Seed'].max():.0f}")
    print(f"  Seed_Diff range: {features_df['Seed_Diff'].min():.0f} - {features_df['Seed_Diff'].max():.0f}")
    
    # Check correlation between seeds and outcome
    seed_outcome_corr = features_df['Seed_Diff'].corr(features_df['Outcome'])
    print(f"\nCorrelation between Seed_Diff and Outcome: {seed_outcome_corr:.4f}")
    
    if abs(seed_outcome_corr) < 0.1:
        print("  [CRITICAL] Seed has very low correlation with outcome")
        print("     This is wrong for March Madness - seeds should be predictive")
    elif abs(seed_outcome_corr) < 0.3:
        print("  [WARNING] Seed correlation is lower than expected")
        print("     Expected: > 0.3 for March Madness")
    else:
        print("  [OK] Seed correlation looks reasonable")
    
    # Test seed-only baseline
    print("\n" + "="*70)
    print("[6/7] SEED-ONLY BASELINE MODEL")
    print("="*70)
    
    # Split data
    train_df = features_df[features_df['Season'] <= 2020]
    val_df = features_df[features_df['Season'].isin([2021, 2022, 2023])]
    
    seed_features = ['Team1_Seed', 'Team2_Seed', 'Seed_Diff']
    X_train_seeds = train_df[seed_features]
    y_train = train_df['Outcome']
    X_val_seeds = val_df[seed_features]
    y_val = val_df['Outcome']
    
    # Train seed-only model
    seed_model = LogisticRegression(max_iter=1000, random_state=42)
    seed_model.fit(X_train_seeds, y_train)
    
    # Evaluate
    train_pred = seed_model.predict_proba(X_train_seeds)[:, 1]
    val_pred = seed_model.predict_proba(X_val_seeds)[:, 1]
    
    train_loss = log_loss(y_train, train_pred)
    val_loss = log_loss(y_val, val_pred)
    val_acc = accuracy_score(y_val, (val_pred >= 0.5).astype(int))
    
    print(f"\nSeed-only model performance:")
    print(f"  Train Log Loss: {train_loss:.4f}")
    print(f"  Val Log Loss:   {val_loss:.4f}")
    print(f"  Val Accuracy:   {val_acc:.4f}")
    
    if val_loss > 0.65:
        print("\n  [CRITICAL] Seed-only model is performing poorly")
        print("     Expected: 0.55-0.60 log loss")
        print("     Possible issues:")
        print("       - Seeds not extracted correctly")
        print("       - Outcome labels might be wrong")
        print("       - Data leakage or other data quality issue")
    elif val_loss > 0.60:
        print("\n  [WARNING] Seed-only model is below expectations")
        print("     Expected: 0.55-0.60 log loss")
    else:
        print("\n  [OK] Seed-only model performance is good")
        print("    This is a reasonable baseline to beat")
    
    # Check train vs val performance
    print("\n" + "="*70)
    print("[7/7] OVERFITTING ANALYSIS")
    print("="*70)
    
    # Train a simple model on all features
    all_features = [c for c in features_df.columns if c not in ['Season', 'Team1', 'Team2', 'Outcome']]
    X_train_all = train_df[all_features]
    X_val_all = val_df[all_features]
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_all, y_train)
    
    train_pred_all = lr_model.predict_proba(X_train_all)[:, 1]
    val_pred_all = lr_model.predict_proba(X_val_all)[:, 1]
    
    train_loss_all = log_loss(y_train, train_pred_all)
    val_loss_all = log_loss(y_val, val_pred_all)
    
    print(f"\nLogistic Regression (all features):")
    print(f"  Train Log Loss: {train_loss_all:.4f}")
    print(f"  Val Log Loss:   {val_loss_all:.4f}")
    print(f"  Difference:     {val_loss_all - train_loss_all:.4f}")
    
    gap = val_loss_all - train_loss_all
    if gap > 0.1:
        print(f"\n  [WARNING] Large train-val gap ({gap:.4f}) suggests overfitting")
    elif gap > 0.05:
        print(f"\n  [WARNING] Moderate train-val gap ({gap:.4f})")
    else:
        print(f"\n  [OK] Train-val gap is reasonable ({gap:.4f})")
    
    # Feature importance check
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE (Logistic Regression Coefficients)")
    print("="*70)
    
    coef_df = pd.DataFrame({
        'Feature': all_features,
        'Coefficient': lr_model.coef_[0]
    })
    coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('Abs_Coef', ascending=False)
    
    print("\nTop 10 most important features:")
    for i, row in coef_df.head(10).iterrows():
        print(f"  {row['Feature']:<25} {row['Coefficient']:>8.4f}")
    
    # Check if seeds are in top 5
    seed_features_in_top5 = any(row['Feature'] in ['Team1_Seed', 'Team2_Seed', 'Seed_Diff'] 
                                  for _, row in coef_df.head(5).iterrows())
    
    if not seed_features_in_top5:
        print("\n  [WARNING] No seed features in top 5")
        print("     For March Madness, seeds should be highly important")
    
    # Summary and recommendations
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    issues = []
    
    if len(val_2023) < 100:
        issues.append("[CRITICAL] Validation set too small (expand to 2021-2023)")
    
    if val_loss > 0.60:
        issues.append("[CRITICAL] Seed-only baseline performing poorly (check seed extraction)")
    
    if not seed_features_in_top5:
        issues.append("[WARNING] Seed features not in top 5 importance")
    
    if gap > 0.1:
        issues.append("[WARNING] Significant overfitting detected")
    
    if val_loss_all > 0.65:
        issues.append("[WARNING] Overall model performance below expectations")
    
    if issues:
        print("\n[ISSUES FOUND]")
        for issue in issues:
            print(f"  {issue}")
        
        print("\n[RECOMMENDED ACTIONS] (in order):")
        print("  1. Expand validation set to 2021-2023")
        print("  2. Verify seed extraction is correct")
        print("  3. Add train performance tracking to ml_pipeline.py")
        print("  4. Simplify tree models (reduce depth/estimators)")
        print("  5. Add cross-validation")
        print("  6. Then proceed with feature engineering")
        print("  7. Finally: hyperparameter tuning")
    else:
        print("\n[SUCCESS] No major issues detected!")
        print("   You can proceed with model improvement")
    
    print("\n" + "="*70)
    print("For detailed analysis, see: NEXT_STEPS_ANALYSIS.md")
    print("="*70)
    
    return {
        'validation_size': len(val_2023),
        'seed_baseline_logloss': val_loss,
        'lr_baseline_logloss': val_loss_all,
        'train_val_gap': gap,
        'seed_correlation': seed_outcome_corr,
        'issues': issues
    }

if __name__ == "__main__":
    results = run_diagnostics()

