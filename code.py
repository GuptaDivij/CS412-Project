from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from datetime import datetime

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")

class MarchMadnessPreprocessor:
  def __init__(self, data_dir):
    self.data_dir = Path(data_dir)
    self.data = {}
    self.features = None
    self.target = None
    self.season_stats = None

  def load_data(self):
    self.data['teams'] = pd.read_csv(self.data_dir / 'MTeams.csv')
    self.data['regular_season'] = pd.read_csv(self.data_dir / 'MRegularSeasonCompactResults.csv')
    self.data['tourney_results'] = pd.read_csv(self.data_dir / 'MNCAATourneyCompactResults.csv')
    self.data['tourney_seeds'] = pd.read_csv(self.data_dir / 'MNCAATourneySeeds.csv')
    self.data['seasons'] = pd.read_csv(self.data_dir / 'MSeasons.csv')
    self.data['regular_season_detailed'] = pd.read_csv(self.data_dir / 'MRegularSeasonDetailedResults.csv')
    self.data['massey_ordinals'] = pd.read_csv(self.data_dir / 'MMasseyOrdinals.csv')
    self.data['team_conferences'] = pd.read_csv(self.data_dir / 'MTeamConferences.csv')
    self.data['conference_tourney'] = pd.read_csv(self.data_dir / 'MConferenceTourneyGames.csv')
    print('load data complete')
    return True

  def analyze_missing_values(self):
    for name, df in self.data.items():
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        if missing.sum() > 0:
            print(f"\n{name.upper()}:")
            print(f"  Total rows: {len(df):,}")
            missing_df = pd.DataFrame({
                'Missing Count': missing[missing > 0],
                'Percentage': missing_pct[missing > 0]
            })
            print(missing_df)
    return True

  def clean_data(self, verbose=True):
    """
    Minimal, table-agnostic cleaner:
    - makes a clean copy of every loaded table
    - trims strings (object cols) and converts empty strings to NA
    - replaces non-finite values (±inf) with NA
    - drops rows with ANY NA
    - drops exact duplicates
    Returns a dict of cleaned DataFrames and stores it in self.cleaned.
    """
    if not self.data:
        raise RuntimeError("No data loaded. Call load_data() first.")

    cleaned = {}
    for name, df in self.data.items():
        d = df.copy()

        # 1) Trim whitespace in ALL object/string columns and normalize empties -> NA
        obj_cols = d.select_dtypes(include=["object"]).columns
        if len(obj_cols) > 0:
            d[obj_cols] = d[obj_cols].apply(lambda s: s.astype(str).str.strip())
            d[obj_cols] = d[obj_cols].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

        # 2) Replace non-finite numeric values (inf/-inf) with NA
        num_cols = d.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            d[num_cols] = d[num_cols].replace([np.inf, -np.inf], pd.NA)

        # 3) Drop rows with ANY NA (fully general)
        before = len(d)
        d = d.dropna(how="any")

        # 4) Drop exact duplicate rows
        d = d.drop_duplicates()

        cleaned[name] = d
        if verbose:
            after = len(d)
            print(f"[{name}] kept {after:,} / {before:,} rows after NA-drop; "
                  f"final {len(d):,} after de-dup.")

    self.cleaned = cleaned
    return cleaned

  def create_season_stats(self):
    rs_data = self.data['regular_season']
    stats_list = []
    for season_year in rs_data['Season'].unique():
      season_data = rs_data[rs_data['Season'] == season_year]
      for team_id in pd.unique(season_data[['WTeamID', 'LTeamID']].values.ravel()):
        wins = season_data[season_data['WTeamID'] == team_id]
        losses = season_data[season_data['LTeamID'] == team_id]
        total_games = len(wins) + len(losses)
        if total_games == 0:
            continue

        win_pct = len(wins) / total_games
        points_for = wins['WScore'].sum() + losses['LScore'].sum()
        points_against = wins['LScore'].sum() + losses['WScore'].sum()
        avg_points_for = points_for / total_games if total_games > 0 else 0
        avg_points_against = points_against / total_games if total_games > 0 else 0

        # Score differential
        score_diff = (wins['WScore'] - wins['LScore']).sum() - (losses['WScore'] - losses['LScore']).sum()
        avg_score_diff = score_diff / total_games if total_games > 0 else 0

        # Home/Away performance
        home_wins = len(wins[wins['WLoc'].isin(['H', 'H1', 'H2'])])
        away_wins = len(wins[wins['WLoc'] == 'A'])
        neutral_wins = len(wins[wins['WLoc'] == 'N'])

        stats_list.append({
          'Season': season_year,
          'TeamID': team_id,
          'Games': total_games,
          'Wins': len(wins),
          'Losses': len(losses),
          'WinPct': win_pct,
          'PointsFor': points_for,
          'PointsAgainst': points_against,
          'AvgPointsFor': avg_points_for,
          'AvgPointsAgainst': avg_points_against,
          'AvgScoreDiff': avg_score_diff,
          'HomeWins': home_wins,
          'AwayWins': away_wins,
          'NeutralWins': neutral_wins,
        })

    season_stats = pd.DataFrame(stats_list)
    self.season_stats = season_stats
    return season_stats

  def parse_seed(self, seed_str):
    """
    Parse seed string like 'W01' or 'X16a' into numeric seed (1-16).
    Returns None if invalid.
    """
    if pd.isna(seed_str) or not isinstance(seed_str, str):
        return None
    # Remove region letter and any suffix (a/b for play-in)
    seed_num = ''.join(c for c in seed_str[1:] if c.isdigit())
    try:
        return int(seed_num)
    except:
        return None

  def get_team_seed(self, season, team_id):
    """Get tournament seed for a team in a given season."""
    seeds = self.data['tourney_seeds']
    seed_row = seeds[(seeds['Season'] == season) & (seeds['TeamID'] == team_id)]
    if len(seed_row) == 0:
        return None
    return self.parse_seed(seed_row.iloc[0]['Seed'])

  def create_features(self):
    matchup_features = []
    tr = self.data['tourney_results']

    for _, game in tr.iterrows():
        season = int(game['Season'])
        w = int(game['WTeamID'])
        l = int(game['LTeamID'])

        # Keep original Team1 = winner, Team2 = loser stats lookup (as you had)
        team1 = w
        team2 = l

        t1_stats = self.season_stats[(self.season_stats['Season'] == season) &
                                     (self.season_stats['TeamID'] == team1)]
        t2_stats = self.season_stats[(self.season_stats['Season'] == season) &
                                     (self.season_stats['TeamID'] == team2)]
        if len(t1_stats) == 0 or len(t2_stats) == 0:
            continue
        t1_stats = t1_stats.iloc[0]
        t2_stats = t2_stats.iloc[0]

        # Get seeds (IMPORTANT: this is what you were missing!)
        t1_seed = self.get_team_seed(season, team1)
        t2_seed = self.get_team_seed(season, team2)

        # Label must be 1 if LOWER TeamID won, else 0
        low = min(w, l)
        outcome = 1 if low == w else 0

        features = {
            'Season': season,
            'Team1': team1,  # winner ID (unchanged)
            'Team2': team2,  # loser ID (unchanged)

            'Team1_WinPct': t1_stats['WinPct'],
            'Team2_WinPct': t2_stats['WinPct'],
            'WinPct_Diff': t1_stats['WinPct'] - t2_stats['WinPct'],
            
            'Team1_AvgPointsFor': t1_stats['AvgPointsFor'],
            'Team2_AvgPointsFor': t2_stats['AvgPointsFor'],
            'PointsFor_Diff': t1_stats['AvgPointsFor'] - t2_stats['AvgPointsFor'],
            
            'Team1_AvgPointsAgainst': t1_stats['AvgPointsAgainst'],
            'Team2_AvgPointsAgainst': t2_stats['AvgPointsAgainst'],
            'PointsAgainst_Diff': t1_stats['AvgPointsAgainst'] - t2_stats['AvgPointsAgainst'],
            
            'Team1_AvgScoreDiff': t1_stats['AvgScoreDiff'],
            'Team2_AvgScoreDiff': t2_stats['AvgScoreDiff'],
            'ScoreDiff_Diff': t1_stats['AvgScoreDiff'] - t2_stats['AvgScoreDiff'],
            
            'Team1_Games': t1_stats['Games'],
            'Team2_Games': t2_stats['Games'],

            # NEW: Seed features (critical for March Madness!)
            'Team1_Seed': t1_seed if t1_seed is not None else 16,  # default to 16 if missing
            'Team2_Seed': t2_seed if t2_seed is not None else 16,
            'Seed_Diff': (t1_seed if t1_seed else 16) - (t2_seed if t2_seed else 16),

            'Outcome': outcome,
        }
        matchup_features.append(features)

    return pd.DataFrame(matchup_features)

  def apply_pca(self, X, n_components=None, variance_threshold=0.95, random_state=0, return_models=False):
    """
    Run PCA on numeric columns of X.
    - Standardizes numeric columns with z-score (mean=0, std=1)
    - If n_components is None, chooses the smallest k s.t. cumulative explained variance >= variance_threshold
    - Stores fitted scaler/PCA/columns for later use in self.pca_state
    - Returns a DataFrame of PCs (PC1..PCk). Optionally also returns (scaler, pca).

    Parameters
    ----------
    X : pd.DataFrame
        Input features (may include non-numeric columns; only numeric are PCA'd).
    n_components : int or None
        If provided, fix the number of components. If None, choose by variance_threshold.
    variance_threshold : float
        Target cumulative explained variance (used only when n_components is None).
    random_state : int
        Random seed for PCA reproducibility.
    return_models : bool
        If True, also return (scaler, pca) along with the PCs DataFrame.

    Returns
    -------
    X_pcs : pd.DataFrame
        Principal components (PC1..PCk).
    (optional) scaler, pca : fitted StandardScaler and PCA objects
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    # Select numeric columns for PCA
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found for PCA.")

    # Standardize numeric features
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(X[num_cols].values)

    # If n_components not given, determine k by variance_threshold
    if n_components is None:
        pca_full = PCA(random_state=random_state)
        pca_full.fit(Z)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, variance_threshold) + 1)  # smallest k s.t. cumvar >= threshold
        n_components = max(1, min(k, Z.shape[1]))  # clamp to [1, #features]

    # Fit final PCA with chosen n_components
    pca = PCA(n_components=n_components, random_state=random_state)
    PCs = pca.fit_transform(Z)

    # Build tidy DataFrame for PCs
    pc_cols = [f"PC{i+1}" for i in range(pca.n_components_)]
    X_pcs = pd.DataFrame(PCs, index=X.index, columns=pc_cols)

    # Persist state for later (e.g., transforming validation/test)
    self.pca_state = {
        "num_cols": num_cols,
        "scaler": scaler,
        "pca": pca,
        "pc_cols": pc_cols,
        "explained_variance_ratio_": pca.explained_variance_ratio_,
        "cumulative_variance_": np.cumsum(pca.explained_variance_ratio_),
    }

    if return_models:
        return X_pcs, scaler, pca
    return X_pcs


  def transform_with_pca(self, X_new):
    """
    Apply previously-fitted scaler/PCA to a new DataFrame (same numeric columns).
    Returns a DataFrame with the same PC columns used during fit.
    """
    if not hasattr(self, "pca_state") or self.pca_state is None:
        raise RuntimeError("No PCA state found. Call apply_pca(...) first.")

    num_cols = self.pca_state["num_cols"]
    scaler = self.pca_state["scaler"]
    pca = self.pca_state["pca"]
    pc_cols = self.pca_state["pc_cols"]

    # Sanity check: ensure required columns exist
    missing = [c for c in num_cols if c not in X_new.columns]
    if missing:
        raise ValueError(f"X_new is missing columns required by PCA: {missing}")

    # Transform
    Z = scaler.transform(X_new[num_cols].values)
    PCs = pca.transform(Z)
    return pd.DataFrame(PCs, index=X_new.index, columns=pc_cols)

  def split_train_test(
    self,
    features_df: pd.DataFrame,
    method: str = "season",                 # "season" or "random"
    train_seasons: list | None = None,      # e.g., list(range(1985, 2023))
    val_seasons: list | None = None,        # e.g., [2023]
    test_size: float = 0.2,                 # used only for method="random"
    random_state: int = 42,
    stratify: bool = True,
    exclude_cols: list | None = None        # columns to exclude from X
):
    """
    Splits features into train/val.
    Returns: X_train, X_val, y_train, y_val, meta_train, meta_val, feature_cols
    where meta_* contains ["Season","Team1","Team2"].

    - method="season": use explicit season lists (train_seasons, val_seasons).
      If not provided, defaults to train <= 2022, val == 2023 (if present).
    - method="random": random split with optional stratification by Outcome.
    """
    if exclude_cols is None:
        exclude_cols = ["Season", "Team1", "Team2", "Outcome"]

    # Basic checks
    for col in ["Season", "Team1", "Team2", "Outcome"]:
        if col not in features_df.columns:
            raise ValueError(f"features_df must contain column '{col}'")

    # Determine feature columns (numeric + any other you want, minus exclude)
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]

    if method == "season":
        # Default seasons if none provided
        if (train_seasons is None) or (val_seasons is None):
            # sensible default: train = <= 2022, val = 2023 (if exists)
            max_train = 2022
            train_seasons = sorted(features_df["Season"].unique())
            train_seasons = [s for s in train_seasons if s <= max_train]
            val_seasons = [s for s in features_df["Season"].unique() if s == (max_train + 1)]
            if len(val_seasons) == 0:
                # fallback: last available season as val
                last = int(features_df["Season"].max())
                val_seasons = [last]
                train_seasons = [s for s in features_df["Season"].unique() if s < last]

        train_mask = features_df["Season"].isin(train_seasons)
        val_mask   = features_df["Season"].isin(val_seasons)

        train_df = features_df.loc[train_mask].copy()
        val_df   = features_df.loc[val_mask].copy()

    elif method == "random":
        # Random split on indices
        y_all = features_df["Outcome"].values
        strat = y_all if (stratify and len(np.unique(y_all)) > 1) else None

        idx_train, idx_val = train_test_split(
            features_df.index,
            test_size=test_size,
            random_state=random_state,
            stratify=strat
        )
        train_df = features_df.loc[idx_train].copy()
        val_df   = features_df.loc[idx_val].copy()
    else:
        raise ValueError("method must be 'season' or 'random'")

    # Build X/y and keep metadata
    X_train = train_df[feature_cols].reset_index(drop=True)
    X_val   = val_df[feature_cols].reset_index(drop=True)
    y_train = train_df["Outcome"].reset_index(drop=True)
    y_val   = val_df["Outcome"].reset_index(drop=True)

    meta_cols = ["Season", "Team1", "Team2"]
    meta_train = train_df[meta_cols].reset_index(drop=True)
    meta_val   = val_df[meta_cols].reset_index(drop=True)

    # Save to self for convenience (optional)
    self.split_state = {
        "method": method,
        "feature_cols": feature_cols,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "train_seasons": train_seasons if method == "season" else None,
        "val_seasons": val_seasons if method == "season" else None,
        "random_state": random_state if method == "random" else None,
        "test_size": test_size if method == "random" else None,
    }

    return X_train, X_val, y_train, y_val, meta_train, meta_val, feature_cols

  def train_and_evaluate_models(self, X_train, X_val, y_train, y_val):
    """
    Train baseline and gradient boosting models, evaluate with log loss.
    Returns dictionary with model names, predictions, and metrics.
    """
    results = {}
    
    print("\n" + "="*60)
    print("MODEL TRAINING & EVALUATION")
    print("="*60)
    
    # 1) Baseline: Logistic Regression
    print("\n[1/4] Training Logistic Regression (Baseline)...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    y_pred_lr = lr.predict_proba(X_val)[:, 1]
    ll_lr = log_loss(y_val, y_pred_lr)
    acc_lr = accuracy_score(y_val, (y_pred_lr >= 0.5).astype(int))
    auc_lr = roc_auc_score(y_val, y_pred_lr)
    
    results['Logistic Regression'] = {
        'model': lr,
        'predictions': y_pred_lr,
        'log_loss': ll_lr,
        'accuracy': acc_lr,
        'auc': auc_lr
    }
    print(f"  Log Loss: {ll_lr:.4f} | Accuracy: {acc_lr:.4f} | AUC: {auc_lr:.4f}")
    
    # 2) XGBoost
    if HAS_XGB:
        print("\n[2/4] Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        
        y_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
        ll_xgb = log_loss(y_val, y_pred_xgb)
        acc_xgb = accuracy_score(y_val, (y_pred_xgb >= 0.5).astype(int))
        auc_xgb = roc_auc_score(y_val, y_pred_xgb)
        
        results['XGBoost'] = {
            'model': xgb_model,
            'predictions': y_pred_xgb,
            'log_loss': ll_xgb,
            'accuracy': acc_xgb,
            'auc': auc_xgb
        }
        print(f"  Log Loss: {ll_xgb:.4f} | Accuracy: {acc_xgb:.4f} | AUC: {auc_xgb:.4f}")
    else:
        print("\n[2/4] XGBoost not available (skipping)")
    
    # 3) LightGBM
    if HAS_LGB:
        print("\n[3/4] Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            random_state=42,
            verbosity=-1
        )
        lgb_model.fit(X_train, y_train)
        
        y_pred_lgb = lgb_model.predict_proba(X_val)[:, 1]
        ll_lgb = log_loss(y_val, y_pred_lgb)
        acc_lgb = accuracy_score(y_val, (y_pred_lgb >= 0.5).astype(int))
        auc_lgb = roc_auc_score(y_val, y_pred_lgb)
        
        results['LightGBM'] = {
            'model': lgb_model,
            'predictions': y_pred_lgb,
            'log_loss': ll_lgb,
            'accuracy': acc_lgb,
            'auc': auc_lgb
        }
        print(f"  Log Loss: {ll_lgb:.4f} | Accuracy: {acc_lgb:.4f} | AUC: {auc_lgb:.4f}")
    else:
        print("\n[3/4] LightGBM not available (skipping)")
    
    # 4) Summary
    print("\n[4/4] RESULTS SUMMARY")
    print("-" * 60)
    print(f"{'Model':<20} {'Log Loss':<12} {'Accuracy':<12} {'AUC':<12}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<20} {res['log_loss']:<12.4f} {res['accuracy']:<12.4f} {res['auc']:<12.4f}")
    print("-" * 60)
    
    # Find best model by log loss
    best_model = min(results.items(), key=lambda x: x[1]['log_loss'])
    print(f"\n🏆 Best Model: {best_model[0]} (Log Loss: {best_model[1]['log_loss']:.4f})")
    
    return results

  def save_results(self, results, X_val, y_val, meta_val, output_dir='output'):
    """
    Save all results to files:
    - Predictions CSV
    - Metrics JSON
    - Trained models (pickle)
    - Feature importance (for tree models)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Clean up old output files (keep only the latest run)
    print("\n🧹 Cleaning up old output files...")
    for old_file in output_path.glob("*_*.csv"):
        old_file.unlink()
    for old_file in output_path.glob("*_*.json"):
        old_file.unlink()
    for old_file in output_path.glob("*_*.pkl"):
        old_file.unlink()
    for old_file in output_path.glob("*_*.txt"):
        old_file.unlink()
    print("✓ Old files removed")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # 1) Save predictions for each model
    predictions_df = meta_val.copy()
    predictions_df['Actual'] = y_val.values
    
    for model_name, res in results.items():
        col_name = model_name.replace(' ', '_')
        predictions_df[f'Pred_{col_name}'] = res['predictions']
    
    pred_file = output_path / f'predictions_{timestamp}.csv'
    predictions_df.to_csv(pred_file, index=False)
    print(f"✓ Saved predictions: {pred_file}")
    
    # 2) Save metrics summary
    metrics = {}
    for model_name, res in results.items():
        metrics[model_name] = {
            'log_loss': float(res['log_loss']),
            'accuracy': float(res['accuracy']),
            'auc': float(res['auc'])
        }
    
    metrics_file = output_path / f'metrics_{timestamp}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics: {metrics_file}")
    
    # 3) Save trained models
    for model_name, res in results.items():
        model_filename = model_name.replace(' ', '_').lower()
        model_file = output_path / f'model_{model_filename}_{timestamp}.pkl'
        
        with open(model_file, 'wb') as f:
            pickle.dump(res['model'], f)
        print(f"✓ Saved model: {model_file}")
    
    # 4) Save feature importance (for tree models)
    for model_name, res in results.items():
        model = res['model']
        
        # XGBoost
        if HAS_XGB and isinstance(model, xgb.XGBClassifier):
            importance = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': X_val.columns,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            imp_file = output_path / f'feature_importance_xgboost_{timestamp}.csv'
            feat_imp_df.to_csv(imp_file, index=False)
            print(f"✓ Saved XGBoost feature importance: {imp_file}")
        
        # LightGBM
        elif HAS_LGB and isinstance(model, lgb.LGBMClassifier):
            importance = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': X_val.columns,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            imp_file = output_path / f'feature_importance_lightgbm_{timestamp}.csv'
            feat_imp_df.to_csv(imp_file, index=False)
            print(f"✓ Saved LightGBM feature importance: {imp_file}")
    
    # 5) Save summary report
    report_file = output_path / f'summary_report_{timestamp}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("NCAA MARCH MADNESS PREDICTION - RESULTS SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Validation Set: {len(y_val)} games\n")
        f.write(f"Validation Seasons: {sorted(meta_val['Season'].unique())}\n\n")
        
        f.write("-"*60 + "\n")
        f.write(f"{'Model':<20} {'Log Loss':<12} {'Accuracy':<12} {'AUC':<12}\n")
        f.write("-"*60 + "\n")
        for model_name, res in results.items():
            f.write(f"{model_name:<20} {res['log_loss']:<12.4f} {res['accuracy']:<12.4f} {res['auc']:<12.4f}\n")
        f.write("-"*60 + "\n\n")
        
        best_model = min(results.items(), key=lambda x: x[1]['log_loss'])
        f.write(f"🏆 Best Model: {best_model[0]}\n")
        f.write(f"   Log Loss: {best_model[1]['log_loss']:.4f}\n")
        f.write(f"   Accuracy: {best_model[1]['accuracy']:.4f}\n")
        f.write(f"   AUC: {best_model[1]['auc']:.4f}\n")
    
    print(f"✓ Saved summary report: {report_file}")
    
    print("\n" + "="*60)
    print(f"All results saved to: {output_path.absolute()}")
    print("="*60)
    
    return output_path


def main():
    """Main execution function"""
    print("="*60)
    print("NCAA March Madness Prediction Pipeline")
    print("="*60)
    
    # Initialize preprocessor with CORRECT path
    preprocessor = MarchMadnessPreprocessor('march-machine-learning-mania-2025')
    
    # 1) Load data
    print("\n[STEP 1/8] Loading data...")
    preprocessor.load_data()
    
    # 2) Clean data
    print("\n[STEP 2/8] Cleaning data...")
    cleaned = preprocessor.clean_data(verbose=True)
    preprocessor.data = preprocessor.cleaned
    
    # 3) Create season statistics
    print("\n[STEP 3/8] Creating season statistics...")
    season_stats = preprocessor.create_season_stats()
    print(f"✓ Season stats shape: {season_stats.shape}")
    
    # 4) Create features
    print("\n[STEP 4/8] Creating matchup features (with SEEDS!)...")
    features_df = preprocessor.create_features()
    print(f"✓ Features shape: {features_df.shape}")
    print(f"✓ Features: {list(features_df.columns)}")
    
    # 5) Season-based split (temporal validation)
    print("\n[STEP 5/8] Splitting data (temporal validation)...")
    X_tr, X_val, y_tr, y_val, meta_tr, meta_val, feat_cols = preprocessor.split_train_test(
        features_df,
        method="season",
        train_seasons=list(range(1985, 2023)),
        val_seasons=[2023]
    )
    print(f"✓ Train: {X_tr.shape}, Val: {X_val.shape}")
    print(f"✓ Train label mean: {y_tr.mean():.3f}, Val label mean: {y_val.mean():.3f}")
    
    # 6) Optional: PCA demo
    print("\n[STEP 6/8] Applying PCA (dimensionality reduction)...")
    feature_cols = [c for c in features_df.columns if c not in ["Season", "Team1", "Team2", "Outcome"]]
    train_df = features_df[features_df["Season"] <= 2022]
    val_df = features_df[features_df["Season"] == 2023]
    
    pcs_train = preprocessor.apply_pca(train_df[feature_cols], variance_threshold=0.95)
    pcs_val = preprocessor.transform_with_pca(val_df[feature_cols])
    
    print(f"✓ PCA Train: {pcs_train.shape}, PCA Val: {pcs_val.shape}")
    print(f"✓ Cumulative variance explained: {preprocessor.pca_state['cumulative_variance_'][-1]:.3f}")
    
    # 7) Train and evaluate models
    print("\n[STEP 7/8] Training and evaluating models...")
    results = preprocessor.train_and_evaluate_models(X_tr, X_val, y_tr, y_val)
    
    # 8) Save all results to files
    print("\n[STEP 8/8] Saving results to files...")
    output_dir = preprocessor.save_results(results, X_val, y_val, meta_val, output_dir='output')
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print(f"\n📁 Results saved in: {output_dir.absolute()}")
    print("\nFiles created:")
    print("  • predictions_YYYYMMDD_HHMMSS.csv      - All model predictions")
    print("  • metrics_YYYYMMDD_HHMMSS.json         - Performance metrics")
    print("  • model_*.pkl                          - Trained models")
    print("  • feature_importance_*.csv             - Feature rankings")
    print("  • summary_report_YYYYMMDD_HHMMSS.txt   - Text summary")
    
    return preprocessor, X_tr, X_val, y_tr, y_val, meta_tr, meta_val, results


if __name__ == "__main__":
    main()