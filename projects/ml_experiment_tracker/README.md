# ML Experiment Tracker

A small learning project for recording machine learning experiments as JSON files
and comparing their results from the command line.

## Project Structure

```text
ml_experiment_tracker/
├── README.md
├── experiments/
│   ├── baseline_logreg.json
│   ├── random_forest.json
│   └── tuned_xgboost.json
└── src/
    └── experiment_tracker.py
```

## Features

- Save experiment metadata, parameters, metrics, and notes as JSON.
- Store each experiment in the `experiments/` folder.
- Compare experiment results in a simple table.
- Run with only the Python standard library.

## Usage

Run commands from this project folder.

### Compare Experiments

```bash
python src/experiment_tracker.py compare
```

### Record a New Experiment

```bash
python src/experiment_tracker.py log \
  --name simple_svm \
  --model SVM \
  --dataset iris \
  --param C=1.0 \
  --param kernel=rbf \
  --metric accuracy=0.956 \
  --metric f1=0.951 \
  --notes "First SVM trial"
```

The command creates a new JSON file in `experiments/`.

## Experiment JSON Format

```json
{
  "id": "baseline_logreg",
  "name": "Baseline Logistic Regression",
  "model": "LogisticRegression",
  "dataset": "iris",
  "created_at": "2026-06-11T09:00:00+09:00",
  "parameters": {
    "max_iter": 200,
    "solver": "lbfgs"
  },
  "metrics": {
    "accuracy": 0.947,
    "f1": 0.944
  },
  "notes": "Simple baseline model."
}
```

