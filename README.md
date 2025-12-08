# NCAA March Madness 2025 Prediction

**Team:** Carlos (carlosd5), Divij Gupta (divij3), Rui Lin (rayl3), Sijie Mei (sijiem2)

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
python ml_pipeline.py
```

This will:
- Load and clean data from `march-machine-learning-mania-2025/`
- Create features and train models
- Save results to `output/` directory

### 3. Generate Report Analysis (Optional)

```bash
python generate_report_analysis.py
```

This generates ablation studies, parameter studies, and baseline comparisons for the final report.

## Data

Data files should be placed in the `march-machine-learning-mania-2025/` directory. Download from [Kaggle March Machine Learning Mania 2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025).

## Output

Results are saved to the `output/` directory:
- `predictions_*.csv` - Model predictions
- `metrics_*.json` - Performance metrics
- `model_*.pkl` - Trained models
- `feature_importance_*.csv` - Feature rankings
- `summary_report_*.txt` - Text summary
