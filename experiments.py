import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

credit = pd.read_csv("german.csv", sep=";")
Xc = credit[credit.columns[1:]].to_numpy(float)
yc = credit[credit.columns[0]].to_numpy(int)
X_tr, X_te, y_tr, y_te = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
cv = StratifiedKFold(5, shuffle=True, random_state=42)

rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=42)
hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=3, random_state=42)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
models = {
    "RF (200 деревьев, дефолт)": RandomForestClassifier(n_estimators=200, random_state=42),
    "RF (depth=8, leaf=3)": rf,
    "HistGB (depth=3)": hgb,
    "MLP (32), alpha=1": make_pipeline(StandardScaler(), MLPClassifier((32,), alpha=1.0, max_iter=500, random_state=42)),
    "Voting RF+HistGB+LR": VotingClassifier([("rf", rf), ("hgb", hgb), ("lr", lr)], voting="soft"),
}
print("| Модель | CV ROC-AUC (train) | ROC-AUC test | Accuracy | Precision | Recall | F1 |")
for name, m in models.items():
    cv_auc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    pr = (p >= 0.5).astype(int)
    print(f"| {name} | {cv_auc:.4f} | {roc_auc_score(y_te, p):.4f} | {accuracy_score(y_te, pr):.3f} | "
          f"{precision_score(y_te, pr):.3f} | {recall_score(y_te, pr):.3f} | {f1_score(y_te, pr):.3f} |")