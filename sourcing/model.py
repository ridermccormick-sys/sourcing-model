"""Positive-unlabeled (PU) bagging ranker.

The firm has a short list of positives (portfolio companies and companies it
pursued), optionally a few explicit negatives (passed because off-thesis), and a
large unlabeled universe. Most unlabeled companies are poor fits, but some are
great ones nobody has looked at yet. Treating every unlabeled company as a
negative would teach the model to downrank exactly the companies we want to
find.

PU bagging (Mordelet & Vert, 2014): in each bag, train positives against a small
random sample of unlabeled companies. Hidden positives in the sample are a
minority, so they add little noise, and averaging many bags gives a stable
ranking.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class _Blend:
    """Soft blend of gradient boosting (non-linear interactions) and a regularised
    logistic regression (stable with tiny training sets)."""

    def __init__(self, kind: str = "blend", random_state: int = 0):
        self.kind = kind
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        self.models_ = []
        if self.kind in ("gbm", "blend"):
            gbm = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3,
                                                 min_samples_leaf=5, l2_regularization=1.0,
                                                 random_state=self.random_state)
            gbm.fit(X, y, sample_weight=sample_weight)
            self.models_.append(gbm)
        if self.kind in ("logreg", "blend"):
            lr = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                           ("scale", StandardScaler()),
                           ("clf", LogisticRegression(C=0.3, max_iter=3000))])
            lr.fit(X, y, clf__sample_weight=sample_weight)
            self.models_.append(lr)
        if not self.models_:
            raise ValueError(f"unknown classifier kind: {self.kind}")
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)


class PUBagger:
    def __init__(self, n_bags: int = 40, unlabeled_ratio: float = 3.0, negative_weight: float = 2.0,
                 classifier: str = "blend", random_state: int = 0):
        self.n_bags = n_bags
        self.unlabeled_ratio = unlabeled_ratio
        self.negative_weight = negative_weight
        self.classifier = classifier
        self.random_state = random_state

    def fit(self, X, y):
        """y: 1 = positive, 0 = explicit negative, -1 = unlabeled."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        P, N, U = np.where(y == 1)[0], np.where(y == 0)[0], np.where(y == -1)[0]
        if len(P) < 5:
            raise ValueError(f"need at least 5 positive seed companies, got {len(P)}")
        if len(U) == 0:
            raise ValueError("no unlabeled companies to learn against")
        rng = np.random.default_rng(self.random_state)
        k = int(min(len(U), max(self.unlabeled_ratio * len(P), 50)))

        self.models_ = []
        oob_sum, oob_cnt = np.zeros(len(y)), np.zeros(len(y))
        for b in range(self.n_bags):
            ub = rng.choice(U, size=k, replace=False)
            idx = np.concatenate([P, N, ub])
            yy = np.concatenate([np.ones(len(P)), np.zeros(len(N) + len(ub))])
            w_neg = np.concatenate([np.full(len(N), self.negative_weight), np.ones(len(ub))])
            w_pos = np.full(len(P), w_neg.sum() / len(P))  # balance classes
            clf = _Blend(self.classifier, random_state=self.random_state + b)
            clf.fit(X[idx], yy, sample_weight=np.concatenate([w_pos, w_neg]))
            self.models_.append(clf)

            oob = np.setdiff1d(U, ub, assume_unique=True)
            oob_sum[oob] += clf.predict_proba(X[oob])[:, 1]
            oob_cnt[oob] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            self.oob_scores_ = np.where(oob_cnt > 0, oob_sum / oob_cnt, np.nan)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.mean([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)


def cross_validate_pu(X, y, n_folds: int = 5, random_state: int = 0, **params) -> dict:
    """Hide a fold of positives inside the unlabeled pool, retrain, and measure where
    they land in the ranking. This tests what matters in production: whether a
    great company nobody has labeled yet gets surfaced near the top.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    P, N, U = np.where(y == 1)[0], np.where(y == 0)[0], np.where(y == -1)[0]
    folds = np.array_split(rng.permutation(P), min(n_folds, len(P)))
    nfolds = np.array_split(rng.permutation(N), min(n_folds, len(N))) if len(N) else [np.array([], int)] * len(folds)

    pos_pct, neg_pct = [], []
    for f, held in enumerate(folds):
        held_n = nfolds[f] if f < len(nfolds) else np.array([], int)
        y_tr = y.copy()
        y_tr[held] = -1      # a hidden positive looks like any other unlabeled company
        y_tr[held_n] = -1
        model = PUBagger(random_state=random_state + 100 * f, **params).fit(X, y_tr)
        u_scores = model.predict(X[U])
        for s in model.predict(X[held]):
            pos_pct.append((u_scores < s).mean())
        for s in (model.predict(X[held_n]) if len(held_n) else []):
            neg_pct.append((u_scores < s).mean())

    pos_pct = np.array(pos_pct)
    out = {
        "n_positives": int(len(P)), "n_negatives": int(len(N)), "n_unlabeled": int(len(U)),
        "heldout_positive_median_percentile": float(np.median(pos_pct)),
        "recall_at_top_5pct": float((pos_pct >= 0.95).mean()),
        "recall_at_top_10pct": float((pos_pct >= 0.90).mean()),
        "recall_at_top_20pct": float((pos_pct >= 0.80).mean()),
    }
    if neg_pct:
        out["heldout_negative_median_percentile"] = float(np.median(neg_pct))
    return out


def explain(model: PUBagger, X: np.ndarray, groups: dict[str, list[int]], background: np.ndarray,
            top_k: int = 3) -> list[tuple[list[tuple[str, float]], tuple[str, float] | None]]:
    """Occlusion attribution: replace each feature group with the universe median and
    measure how much the score drops. Returns (top drivers, biggest drag) per row.
    """
    X = np.asarray(X, dtype=float)
    base = model.predict(X)
    deltas = {}
    for g, cols in groups.items():
        Xm = X.copy()
        Xm[:, cols] = background[cols]
        deltas[g] = base - model.predict(Xm)
    names = list(deltas)
    D = np.column_stack([deltas[g] for g in names])
    out = []
    for i in range(len(X)):
        order = np.argsort(-D[i])
        ups = [(names[j], float(D[i, j])) for j in order[:top_k] if D[i, j] > 0.005]
        j = int(np.argmin(D[i]))
        drag = (names[j], float(D[i, j])) if D[i, j] < -0.005 else None
        out.append((ups, drag))
    return out
