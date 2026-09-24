import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os


def evaluate_model():
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    data = pd.read_csv('data/initial_data.csv')
    pipeline = joblib.load('models/fitted_model.pkl')

    cv_strategy = StratifiedKFold(n_splits=params['n_splits'])

    cv_res = cross_validate(
        pipeline,
        data,
        data[params['target_col']],
        cv=cv_strategy,
        n_jobs=-1,
        scoring=['f1', 'roc_auc']
    )

    metrics = {
        key: round(float(value.mean()), 3)
        for key, value in cv_res.items()
    }

    os.makedirs('cv_results', exist_ok=True)
    with open('cv_results/cv_res.json', 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    evaluate_model()
