import json
import pandas as pd
from sklearn.model_selection import train_test_split

from grader_contracts.ensembles import EnsembleInput
from ensemble_models import run_ensemble_experiment

credit = pd.read_csv("german.csv", sep=";")
target = credit.columns[0]
c_features = list(credit.columns[1:])
Xc = credit[c_features].to_numpy(float)
yc = credit[target].to_numpy(int)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=1, stratify=yc)

res = run_ensemble_experiment(EnsembleInput(Xc_tr, yc_tr, Xc_te, yc_te, random_seed=42))
print(res.model_name)
for name in ("roc_auc", "accuracy", "precision", "recall", "f1"):
    print(f"{name}: {getattr(res, name)}")


with open("experiment_results.json") as f:
    results = json.load(f)
results["ensemble"] = {
    "model_name": res.model_name,
    "roc_auc": res.roc_auc,
    "accuracy": res.accuracy,
    "precision": res.precision,
    "recall": res.recall,
    "f1": res.f1,
}
with open("experiment_results.json", "w") as f:
    json.dump(results, f, indent=2)
