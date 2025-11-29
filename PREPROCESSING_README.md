# March Madness 2025 Preprocessing Guide

This preprocessing pipeline handles data loading, cleaning, feature engineering, and dimensionality reduction for the March Madness prediction challenge.

## Files Created

1. **`preprocessing.py`** - Main preprocessing class with all functionality
2. **`preprocessing_example.py`** - Example usage code for Jupyter notebook
3. **`requirements.txt`** - Required Python packages

## Installation

First, install the required packages:

```bash
pip install pandas numpy scikit-learn
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Usage in Jupyter Notebook

### Quick Start

1. Import the preprocessor:
```python
from preprocessing import MarchMadnessPreprocessor
```

2. Initialize and load data:
```python
preprocessor = MarchMadnessPreprocessor()
preprocessor.load_data()
```

3. Run preprocessing pipeline:
```python
# Analyze missing values
preprocessor.analyze_missing_values()

# Clean data
preprocessor.clean_data()

# Create season statistics
season_stats = preprocessor.create_season_stats()

# Create matchup features
features_df = preprocessor.create_matchup_features(
    season_stats, 
    preprocessor.data.get('tourney_seeds')
)

# Prepare for modeling
X, y = preprocessor.prepare_features_for_modeling(features_df)
```

4. Analyze dimensionality and apply PCA if needed:
```python
# Analyze dimensionality
dim_analysis = preprocessor.analyze_dimensionality(X)

# Apply PCA if recommended
if dim_analysis['pca_recommended']:
    X_pca, pca_model = preprocessor.apply_pca(X, variance_threshold=0.95)
    X_final = X_pca
else:
    X_final = X
```

## Key Features

### 1. Data Loading
- Automatically loads all relevant datasets:
  - Teams, Regular Season Results, Tournament Results
  - Tournament Seeds, Season Info
  - Optional: Detailed Stats, Massey Ordinals, Conferences

### 2. Missing Value Handling
- Analyzes missing values across all datasets
- Fills missing values with appropriate strategies:
  - Seeds: Median imputation
  - Numeric features: Median imputation
  - Handles missing data gracefully

### 3. Data Cleaning
- Removes invalid rows (negative scores, invalid team IDs)
- Reduces noise in datasets
- Filters Massey ordinals to final rankings only
- Validates data integrity

### 4. Feature Engineering
- **Season Statistics**: Aggregates team performance per season
  - Win percentage, points for/against, score differential
  - Home/away/neutral performance
- **Matchup Features**: Creates features for team-vs-team matchups
  - Win percentage differentials
  - Points for/against differentials
  - Score differential
  - Seed information (if available)

### 5. Dimensionality Reduction
- **Analysis**: Checks feature count, correlations, and redundancy
- **PCA**: Optional dimensionality reduction
  - Automatic component selection based on variance threshold
  - Manual component specification
  - Standardization before PCA
  - Variance explained reporting

## Relevant Datasets Identified

### Core Datasets (Essential)
1. **MRegularSeasonCompactResults.csv** - Regular season game results
2. **MNCAATourneyCompactResults.csv** - Historical tournament results
3. **MNCAATourneySeeds.csv** - Tournament seeds (important feature)
4. **MTeams.csv** - Team metadata

### Optional Datasets (Valuable)
5. **MRegularSeasonDetailedResults.csv** - Detailed game statistics
6. **MMasseyOrdinals.csv** - Ranking systems (can add features)
7. **MTeamConferences.csv** - Conference information
8. **MConferenceTourneyGames.csv** - Conference tournament games

## Output Features

The preprocessing creates the following features for each matchup:

- `Team1_WinPct`, `Team2_WinPct`, `WinPct_Diff`
- `Team1_AvgPointsFor`, `Team2_AvgPointsFor`, `PointsFor_Diff`
- `Team1_AvgPointsAgainst`, `Team2_AvgPointsAgainst`, `PointsAgainst_Diff`
- `Team1_AvgScoreDiff`, `Team2_AvgScoreDiff`, `ScoreDiff_Diff`
- `Team1_Games`, `Team2_Games`
- `Team1_Seed`, `Team2_Seed`, `Seed_Diff` (if available)

## Feature Visualization 🎨

The preprocessing class includes comprehensive visualization tools to identify useful features:

### Quick Visualization
```python
# Run all visualizations at once
preprocessor.visualize_all(X, y)
```

### Individual Visualizations

1. **Feature Correlation Heatmap** - Identify redundant features
   ```python
   preprocessor.plot_feature_correlation(X)
   ```

2. **Feature Importance** - See which features matter most
   ```python
   preprocessor.plot_feature_importance(X, y, method='random_forest')
   ```

3. **Feature Distributions** - Compare distributions for wins vs losses
   ```python
   preprocessor.plot_feature_distributions(X, y)
   ```

4. **Feature vs Target** - Box plots showing feature differences
   ```python
   preprocessor.plot_feature_vs_target(X, y)
   ```

5. **PCA Variance Analysis** - Understand dimensionality reduction
   ```python
   preprocessor.plot_pca_variance(X)
   ```

See `VISUALIZATION_GUIDE.md` for detailed usage and interpretation guide.

## Next Steps

After preprocessing:

1. **Visualize Features**: Use visualization tools to identify useful features
2. **Feature Selection**: Remove redundant or unimportant features
3. **Split Data**: Train/validation/test splits
4. **Train Models**: Logistic Regression, Random Forest, XGBoost, Neural Networks
5. **Evaluate**: Cross-validation, performance metrics
6. **Generate Predictions**: For 2025 tournament matchups

## Example Workflow

See `preprocessing_example.py` for a complete example with all cells ready to copy into your Jupyter notebook.

## Notes

- The script handles missing data gracefully
- PCA is recommended when feature count > 10
- All features are numeric and ready for machine learning models
- The preprocessing is designed to be modular - you can use individual functions as needed

