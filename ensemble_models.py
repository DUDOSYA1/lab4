from __future__ import annotations

import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target).ravel()
    X_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target).ravel()

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    seed = data.random_seed

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=1
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=seed
    )
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


    model = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("hgb", hgb),
            ("lr", lr)
        ],
        voting="soft"
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return EnsembleResult(
        model_name="Soft Voting",
        probabilities= probabilities,
        predictions= predictions,
        roc_auc= float(roc_auc_score(y_test, probabilities)),
        accuracy= float(accuracy_score(y_test, predictions)),
        precision= float(precision_score(y_test, predictions)),
        recall= float(recall_score(y_test, predictions)),
        f1= float(f1_score(y_test, predictions)),
    )