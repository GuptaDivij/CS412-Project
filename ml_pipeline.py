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
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb

try:
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
except:
    plt.style.use('default')
sns.set_palette("husl")

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

  def clean_data(self, verbose=False):
    if not self.data:
        raise RuntimeError("No data loaded. Call load_data() first.")

    cleaned = {}
    for name, df in self.data.items():
        d = df.copy()
        obj_cols = d.select_dtypes(include=["object"]).columns
        if len(obj_cols) > 0:
            d[obj_cols] = d[obj_cols].apply(lambda s: s.astype(str).str.strip())
            d[obj_cols] = d[obj_cols].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        num_cols = d.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            d[num_cols] = d[num_cols].replace([np.inf, -np.inf], pd.NA)
        d = d.dropna(how="any")
        d = d.drop_duplicates()
        cleaned[name] = d

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

        score_diff = (wins['WScore'] - wins['LScore']).sum() - (losses['WScore'] - losses['LScore']).sum()
        avg_score_diff = score_diff / total_games if total_games > 0 else 0

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
    if pd.isna(seed_str) or not isinstance(seed_str, str):
        return None
    seed_num = ''.join(c for c in seed_str[1:] if c.isdigit())
    try:
        return int(seed_num)
    except:
        return None

  def get_team_seed(self, season, team_id):
    seeds = self.data['tourney_seeds']
    seed_row = seeds[(seeds['Season'] == season) & (seeds['TeamID'] == team_id)]
    if len(seed_row) == 0:
        return None
    return self.parse_seed(seed_row.iloc[0]['Seed'])

  def create_features(self):
    matchup_features = []
    tr = self.data['tourney_results']

    for idx, game in tr.iterrows():
        season = int(game['Season'])
        w = int(game['WTeamID'])
        l = int(game['LTeamID'])

        team1 = min(w, l)
        team2 = max(w, l)
        outcome = 1 if team1 == w else 0

        t1_stats = self.season_stats[(self.season_stats['Season'] == season) &
                                     (self.season_stats['TeamID'] == team1)]
        t2_stats = self.season_stats[(self.season_stats['Season'] == season) &
                                     (self.season_stats['TeamID'] == team2)]
        if len(t1_stats) == 0 or len(t2_stats) == 0:
            continue
        t1_stats = t1_stats.iloc[0]
        t2_stats = t2_stats.iloc[0]

        t1_seed = self.get_team_seed(season, team1)
        t2_seed = self.get_team_seed(season, team2)

        features = {
            'Season': season,
            'Team1': team1,
            'Team2': team2,
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
            'Team1_Seed': t1_seed if t1_seed is not None else 16,
            'Team2_Seed': t2_seed if t2_seed is not None else 16,
            'Seed_Diff': (t1_seed if t1_seed else 16) - (t2_seed if t2_seed else 16),
            'Outcome': outcome,
        }
        matchup_features.append(features)

    return pd.DataFrame(matchup_features)

  def apply_pca(self, X, n_components=None, variance_threshold=0.95, random_state=0, return_models=False):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found for PCA.")

    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(X[num_cols].values)

    if n_components is None:
        pca_full = PCA(random_state=random_state)
        pca_full.fit(Z)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, variance_threshold) + 1)
        n_components = max(1, min(k, Z.shape[1]))

    pca = PCA(n_components=n_components, random_state=random_state)
    PCs = pca.fit_transform(Z)

    pc_cols = [f"PC{i+1}" for i in range(pca.n_components_)]
    X_pcs = pd.DataFrame(PCs, index=X.index, columns=pc_cols)

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
    if not hasattr(self, "pca_state") or self.pca_state is None:
        raise RuntimeError("No PCA state found. Call apply_pca(...) first.")

    num_cols = self.pca_state["num_cols"]
    scaler = self.pca_state["scaler"]
    pca = self.pca_state["pca"]
    pc_cols = self.pca_state["pc_cols"]

    missing = [c for c in num_cols if c not in X_new.columns]
    if missing:
        raise ValueError(f"X_new is missing columns required by PCA: {missing}")

    Z = scaler.transform(X_new[num_cols].values)
    PCs = pca.transform(Z)
    return pd.DataFrame(PCs, index=X_new.index, columns=pc_cols)

  def split_train_test(
    self,
    features_df,
    method="season",
    train_seasons=None,
    val_seasons=None,
    test_size=0.2,
    random_state=42,
    stratify=True,
    exclude_cols=None
):
    if exclude_cols is None:
        exclude_cols = ["Season", "Team1", "Team2", "Outcome"]

    for col in ["Season", "Team1", "Team2", "Outcome"]:
        if col not in features_df.columns:
            raise ValueError(f"features_df must contain column '{col}'")

    feature_cols = [c for c in features_df.columns if c not in exclude_cols]

    if method == "season":
        if (train_seasons is None) or (val_seasons is None):
            max_train = 2022
            train_seasons = sorted(features_df["Season"].unique())
            train_seasons = [s for s in train_seasons if s <= max_train]
            val_seasons = [s for s in features_df["Season"].unique() if s == (max_train + 1)]
            if len(val_seasons) == 0:
                last = int(features_df["Season"].max())
                val_seasons = [last]
                train_seasons = [s for s in features_df["Season"].unique() if s < last]

        train_mask = features_df["Season"].isin(train_seasons)
        val_mask   = features_df["Season"].isin(val_seasons)
        train_df = features_df.loc[train_mask].copy()
        val_df   = features_df.loc[val_mask].copy()

    elif method == "random":
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

    X_train = train_df[feature_cols].reset_index(drop=True)
    X_val   = val_df[feature_cols].reset_index(drop=True)
    y_train = train_df["Outcome"].reset_index(drop=True)
    y_val   = val_df["Outcome"].reset_index(drop=True)

    meta_cols = ["Season", "Team1", "Team2"]
    meta_train = train_df[meta_cols].reset_index(drop=True)
    meta_val   = val_df[meta_cols].reset_index(drop=True)

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
    results = {}
    
    print("Training models...")
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_val_pred_lr = lr.predict_proba(X_val)[:, 1]
    results['Logistic Regression'] = {
        'model': lr,
        'predictions': y_val_pred_lr,
        'log_loss': log_loss(y_val, y_val_pred_lr),
        'accuracy': accuracy_score(y_val, (y_val_pred_lr >= 0.5).astype(int)),
        'auc': roc_auc_score(y_val, y_val_pred_lr)
    }
    
    xgb_model = xgb.XGBClassifier(
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
    xgb_model.fit(X_train, y_train)
    y_val_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
    results['XGBoost'] = {
        'model': xgb_model,
        'predictions': y_val_pred_xgb,
        'log_loss': log_loss(y_val, y_val_pred_xgb),
        'accuracy': accuracy_score(y_val, (y_val_pred_xgb >= 0.5).astype(int)),
        'auc': roc_auc_score(y_val, y_val_pred_xgb)
    }
    
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        learning_rate=0.05,
        max_depth=3,
        n_estimators=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    y_val_pred_lgb = lgb_model.predict_proba(X_val)[:, 1]
    results['LightGBM'] = {
        'model': lgb_model,
        'predictions': y_val_pred_lgb,
        'log_loss': log_loss(y_val, y_val_pred_lgb),
        'accuracy': accuracy_score(y_val, (y_val_pred_lgb >= 0.5).astype(int)),
        'auc': roc_auc_score(y_val, y_val_pred_lgb)
    }
    
    print(f"{'Model':<20} {'Log Loss':<12} {'Accuracy':<12} {'AUC':<12}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<20} {res['log_loss']:<12.4f} {res['accuracy']:<12.4f} {res['auc']:<12.4f}")
    
    best_model = min(results.items(), key=lambda x: x[1]['log_loss'])
    print(f"\nBest Model: {best_model[0]} (Log Loss: {best_model[1]['log_loss']:.4f})")
    
    return results

  def perform_eda(self, features_df, season_stats, output_dir='eda'):
    eda_path = Path(output_dir)
    eda_path.mkdir(exist_ok=True)
    
    feature_cols = [c for c in features_df.columns if c not in ['Season', 'Team1', 'Team2', 'Outcome']]
    numeric_features = [c for c in feature_cols if features_df[c].dtype in [np.int64, np.float64]]
    
    # target variable distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    outcome_counts = features_df['Outcome'].value_counts()
    axes[0].bar(outcome_counts.index, outcome_counts.values)
    axes[0].set_xlabel('Outcome')
    axes[0].set_ylabel('Count')
    axes[0].set_xticks([0, 1])
    outcome_by_season = features_df.groupby('Season')['Outcome'].mean()
    axes[1].plot(outcome_by_season.index, outcome_by_season.values)
    axes[1].set_xlabel('Season')
    axes[1].set_ylabel('Mean Outcome')
    plt.tight_layout()
    plt.savefig(eda_path / '01_target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # feature distributions
    n_features = len(numeric_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    for i, feat in enumerate(numeric_features[:len(axes)]):
        axes[i].hist(features_df[feat].dropna(), bins=30)
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel('Frequency')
    for i in range(len(numeric_features), len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(eda_path / '02_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # feature correlation heatmap
    corr_matrix = features_df[numeric_features].corr()
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.tight_layout()
    plt.savefig(eda_path / '03_feature_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # seed analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if 'Team1_Seed' in features_df.columns and 'Team2_Seed' in features_df.columns:
        seed_data = features_df[['Team1_Seed', 'Team2_Seed']].melt()
        axes[0, 0].hist(seed_data['value'].dropna(), bins=16)
    axes[0, 0].set_xlabel('Seed')
    axes[0, 0].set_ylabel('Frequency')
    seed_diff_outcome = features_df.groupby('Seed_Diff')['Outcome'].mean()
    axes[0, 1].bar(seed_diff_outcome.index, seed_diff_outcome.values)
    axes[0, 1].set_xlabel('Seed Difference')
    axes[0, 1].set_ylabel('Win Rate')
    seed1_winrate = features_df.groupby('Team1_Seed')['Outcome'].mean()
    seed2_winrate = features_df.groupby('Team2_Seed')['Outcome'].mean()
    axes[1, 0].plot(seed1_winrate.index, seed1_winrate.values, label='Team1')
    axes[1, 0].plot(seed2_winrate.index, seed2_winrate.values, label='Team2')
    axes[1, 0].set_xlabel('Seed')
    axes[1, 0].set_ylabel('Win Rate')
    axes[1, 0].legend()
    axes[1, 1].hist(features_df['Seed_Diff'].dropna(), bins=30)
    axes[1, 1].set_xlabel('Seed Difference')
    axes[1, 1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(eda_path / '04_seed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # season statistics analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(season_stats['WinPct'].dropna(), bins=50)
    axes[0, 0].set_xlabel('Win Percentage')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 1].hist(season_stats['AvgScoreDiff'].dropna(), bins=50)
    axes[0, 1].set_xlabel('Score Difference')
    axes[0, 1].set_ylabel('Frequency')
    sample_stats = season_stats.sample(min(5000, len(season_stats)))
    axes[1, 0].scatter(sample_stats['AvgPointsFor'], sample_stats['AvgPointsAgainst'], alpha=0.3, s=10)
    axes[1, 0].set_xlabel('Points For')
    axes[1, 0].set_ylabel('Points Against')
    winpct_by_season = season_stats.groupby('Season')['WinPct'].mean()
    axes[1, 1].plot(winpct_by_season.index, winpct_by_season.values)
    axes[1, 1].set_xlabel('Season')
    axes[1, 1].set_ylabel('Win Percentage')
    plt.tight_layout()
    plt.savefig(eda_path / '05_season_stats_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # feature vs outcome analysis
    key_features = ['WinPct_Diff', 'ScoreDiff_Diff', 'PointsFor_Diff', 'Seed_Diff']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, feat in enumerate(key_features):
        if feat in features_df.columns:
            data_0 = features_df[features_df['Outcome'] == 0][feat].dropna()
            data_1 = features_df[features_df['Outcome'] == 1][feat].dropna()
            axes[i].boxplot([data_0, data_1], tick_labels=['Higher ID', 'Lower ID'])
            axes[i].set_ylabel(feat)
    plt.tight_layout()
    plt.savefig(eda_path / '06_feature_vs_outcome.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # tournament games over time
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    games_per_season = features_df.groupby('Season').size()
    axes[0].bar(games_per_season.index, games_per_season.values)
    axes[0].set_xlabel('Season')
    axes[0].set_ylabel('Games')
    teams_per_season = season_stats.groupby('Season')['TeamID'].nunique()
    axes[1].plot(teams_per_season.index, teams_per_season.values)
    axes[1].set_xlabel('Season')
    axes[1].set_ylabel('Teams')
    plt.tight_layout()
    plt.savefig(eda_path / '07_tournament_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return eda_path

  def save_results(self, results, X_val, y_val, meta_val, output_dir='output'):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for f in output_path.glob("*_*.csv"):
        f.unlink()
    for f in output_path.glob("*_*.json"):
        f.unlink()
    for f in output_path.glob("*_*.pkl"):
        f.unlink()
    for f in output_path.glob("*_*.txt"):
        f.unlink()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    predictions_df = meta_val.copy()
    predictions_df['Actual'] = y_val.values
    for model_name, res in results.items():
        col_name = model_name.replace(' ', '_')
        predictions_df[f'Pred_{col_name}'] = res['predictions']
    predictions_df.to_csv(output_path / f'predictions_{timestamp}.csv', index=False)
    
    metrics = {}
    for model_name, res in results.items():
        metrics[model_name] = {
            'log_loss': float(res['log_loss']),
            'accuracy': float(res['accuracy']),
            'auc': float(res['auc'])
        }
    with open(output_path / f'metrics_{timestamp}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    for model_name, res in results.items():
        model_filename = model_name.replace(' ', '_').lower()
        with open(output_path / f'model_{model_filename}_{timestamp}.pkl', 'wb') as f:
            pickle.dump(res['model'], f)
    
    for model_name, res in results.items():
        model = res['model']
        if isinstance(model, xgb.XGBClassifier):
            feat_imp_df = pd.DataFrame({
                'Feature': X_val.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            feat_imp_df.to_csv(output_path / f'feature_importance_xgboost_{timestamp}.csv', index=False)
        elif isinstance(model, lgb.LGBMClassifier):
            feat_imp_df = pd.DataFrame({
                'Feature': X_val.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            feat_imp_df.to_csv(output_path / f'feature_importance_lightgbm_{timestamp}.csv', index=False)
    
    best_model = min(results.items(), key=lambda x: x[1]['log_loss'])
    with open(output_path / f'summary_report_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write(f"NCAA MARCH MADNESS PREDICTION - RESULTS SUMMARY\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Validation Set: {len(y_val)} games\n\n")
        f.write(f"{'Model':<20} {'Log Loss':<12} {'Accuracy':<12} {'AUC':<12}\n")
        for model_name, res in results.items():
            f.write(f"{model_name:<20} {res['log_loss']:<12.4f} {res['accuracy']:<12.4f} {res['auc']:<12.4f}\n")
        f.write(f"\nBest Model: {best_model[0]}\n")
        f.write(f"Log Loss: {best_model[1]['log_loss']:.4f}\n")
    
    print(f"Results saved to {output_dir}/")
    return output_path


def run_eda_only(data_dir='march-machine-learning-mania-2025', output_dir='eda'):
    preprocessor = MarchMadnessPreprocessor(data_dir)
    preprocessor.load_data()
    preprocessor.clean_data(verbose=False)
    preprocessor.data = preprocessor.cleaned
    season_stats = preprocessor.create_season_stats()
    features_df = preprocessor.create_features()
    eda_path = preprocessor.perform_eda(features_df, season_stats, output_dir=output_dir)
    return preprocessor, features_df, season_stats, eda_path


def main():
    print("NCAA March Madness Prediction Pipeline")
    
    preprocessor = MarchMadnessPreprocessor('march-machine-learning-mania-2025')
    print("Loading data...")
    preprocessor.load_data()
    
    print("Cleaning data...")
    preprocessor.clean_data(verbose=False)
    preprocessor.data = preprocessor.cleaned
    
    print("Creating features...")
    season_stats = preprocessor.create_season_stats()
    features_df = preprocessor.create_features()
    
    print("Splitting data...")
    X_tr, X_val, y_tr, y_val, meta_tr, meta_val, feat_cols = preprocessor.split_train_test(
        features_df,
        method="season",
        train_seasons=list(range(1985, 2021)),
        val_seasons=[2021, 2022, 2023]
    )
    
    print("Applying PCA...")
    feature_cols = [c for c in features_df.columns if c not in ["Season", "Team1", "Team2", "Outcome"]]
    train_df = features_df[features_df["Season"] <= 2020]
    val_df = features_df[features_df["Season"].isin([2021, 2022, 2023])]
    pcs_train = preprocessor.apply_pca(train_df[feature_cols], variance_threshold=0.95)
    pcs_val = preprocessor.transform_with_pca(val_df[feature_cols])
    
    results = preprocessor.train_and_evaluate_models(X_tr, X_val, y_tr, y_val)
    preprocessor.save_results(results, X_val, y_val, meta_val, output_dir='output')
    
    print("Pipeline complete!")
    return preprocessor, X_tr, X_val, y_tr, y_val, meta_tr, meta_val, results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == '--eda':
        run_eda_only()
    else:
        main()